import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Additive attention pooling (Lin et al., 2017-style: a small 2-layer
    scoring MLP with tanh, softmax over the sequence dimension, weighted
    sum). Replaces flatten + a large FC: instead of treating all CNN
    time steps as one fixed concatenated vector, the model learns
    a per-sample weight for each time step - i.e. which motif regions
    actually matter for this CpG - before collapsing to a fixed-size
    vector. The weights are also interpretable (see last_attention_weights
    on DanQ_Sequence). Length-agnostic (works over whatever L the encoder
    upstream produces - see DanQ_Sequence's use_multiscale_cnn, which
    changes L from the original 36).

    Input:  [B, L, D]
    Output: [B, D]
    """

    def __init__(self, input_dim: int, attention_hidden_dim: int):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(input_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(sequence)  # [B, L, 1]
        weights = torch.softmax(scores, dim=1)  # softmax over L (positions), not features
        pooled = (sequence * weights).sum(dim=1)  # [B, D]
        return pooled, weights.squeeze(-1)  # weights: [B, L], for interpretability


SEQUENCE_LENGTH = 501
TOTAL_CNN_CHANNELS = 320  # matches the original single-branch DanQ conv1's out_channels

# Parallel kernel sizes for use_multiscale_cnn=True - short (core TF-binding-
# motif scale) through the original DanQ kernel_size=26 (kept as one of the
# branches, not replaced, so the model can still fall back to exactly what
# worked before) up to a wider-context branch. Channels split evenly across
# branches so TOTAL_CNN_CHANNELS - and everything downstream (BiLSTM
# input_size, attention_pool width) - stays unchanged from the single-branch
# path; only the CNN front end differs.
MULTISCALE_KERNEL_SIZES = (8, 16, 26, 34)
MULTISCALE_CHANNELS_PER_BRANCH = TOTAL_CNN_CHANNELS // len(MULTISCALE_KERNEL_SIZES)
assert MULTISCALE_CHANNELS_PER_BRANCH * len(MULTISCALE_KERNEL_SIZES) == TOTAL_CNN_CHANNELS

NUM_CNN_POSITIONS = 36  # (501 - 26 + 1) // 13 - single-branch path (use_multiscale_cnn=False)


class DanQ_Sequence(nn.Module):
    """
    DanQ-based 1D DNA sequence branch adapted from ncVarPred.

    Input:
        [batch_size, 4, 501]

    Output:
        [batch_size, 925]

    use_self_attention=False (default) reproduces the original ncVarPred
    DanQ path bit-for-bit: Conv1d -> MaxPool -> BiLSTM -> attention pooling.

    use_self_attention=True replaces the BiLSTM with a lightweight
    Transformer encoder (self-attention) over the CNN positions - cheap
    (negligible next to the graph branch's cost), and lets positions attend
    directly to any other position instead of only through a sequential
    BiLSTM bottleneck. Since self-attention has no inherent notion of
    order, a learned positional embedding is added first. This is an
    ablation toggle: both paths are kept so the two can be trained and
    compared under the exact same script. Compared once already
    (ablation_attention_sequence.json, GM12878): BiLSTM reached a better
    best_val_loss (0.518 vs 0.584) and higher val_mcc at that point (0.651
    vs 0.641) than self-attention - but that run used a fixed LR with no
    warmup/scheduler for either variant, and self-attention's validation
    loss oscillated in a way consistent with exactly the LR-sensitivity
    Transformers are known for without warmup (Vaswani et al., 2017) - so
    this result shouldn't be read as a clean "BiLSTM is architecturally
    better" conclusion, only as "BiLSTM did better under those specific,
    not Transformer-tuned training conditions."

    use_multiscale_cnn=False (default) reproduces the original single
    Conv1d(kernel_size=26) front end. use_multiscale_cnn=True instead runs
    MULTISCALE_KERNEL_SIZES parallel Conv1d branches (each with 'same'
    padding, so every branch's output has the same length regardless of
    its kernel size and they can be concatenated along the channel
    dimension) and concatenates their outputs - short kernels capture
    core-motif-scale patterns, longer ones capture broader context, and
    the model can weight each scale's contribution rather than committing
    to one fixed receptive field. 'same' padding preserves the full 501-bp
    length (vs the original no-padding conv's 476), so the post-maxpool
    position count differs (38 vs 36) - handled automatically (see
    num_cnn_positions below), not hardcoded, so this composes correctly
    with use_self_attention too if both are ever enabled together.
    """

    def __init__(self, use_self_attention: bool = False, use_multiscale_cnn: bool = False):
        super(DanQ_Sequence, self).__init__()

        self.use_self_attention = use_self_attention
        self.use_multiscale_cnn = use_multiscale_cnn

        if self.use_multiscale_cnn:
            self.multiscale_convs = nn.ModuleList([
                nn.Conv1d(
                    in_channels=4,
                    out_channels=MULTISCALE_CHANNELS_PER_BRANCH,
                    kernel_size=kernel_size,
                    padding="same",
                )
                for kernel_size in MULTISCALE_KERNEL_SIZES
            ])
            conv_output_length = SEQUENCE_LENGTH  # 'same' padding preserves length for every branch
        else:
            # Original ncVarPred DanQ convolution.
            self.conv1 = nn.Conv1d(
                in_channels=4,
                out_channels=TOTAL_CNN_CHANNELS,
                kernel_size=26,
            )
            conv_output_length = SEQUENCE_LENGTH - 26 + 1  # 476, no padding

        # Original ncVarPred pooling layer.
        self.maxpool = nn.MaxPool1d(
            kernel_size=13,
            stride=13,
        )
        num_cnn_positions = (conv_output_length - 13) // 13 + 1  # 36 single-branch, 38 multiscale

        self.drop1 = nn.Dropout(
            p=0.2
        )

        self.drop2 = nn.Dropout(
            p=0.5
        )

        if self.use_self_attention:
            # Learned positional embedding: self-attention has no built-in
            # sense of position order, unlike the BiLSTM it replaces. Sized
            # off num_cnn_positions (not the NUM_CNN_POSITIONS module
            # constant), so this stays correct even combined with
            # use_multiscale_cnn's different position count.
            self.positional_embedding = nn.Parameter(
                torch.zeros(num_cnn_positions, TOTAL_CNN_CHANNELS)
            )
            nn.init.normal_(self.positional_embedding, std=0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=TOTAL_CNN_CHANNELS,
                nhead=8,
                dim_feedforward=TOTAL_CNN_CHANNELS * 2,
                dropout=0.1,
                batch_first=True,
            )
            self.self_attention_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=2,
            )

            sequence_encoder_output_dim = TOTAL_CNN_CHANNELS
        else:
            # Original ncVarPred bidirectional LSTM.
            self.bilstm = nn.LSTM(
                input_size=TOTAL_CNN_CHANNELS,
                hidden_size=TOTAL_CNN_CHANNELS,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )

            sequence_encoder_output_dim = TOTAL_CNN_CHANNELS * 2  # forward + backward

        # For a 501 bp sequence (single-branch CNN path):
        #
        # Conv1d:
        # 501 -> 476
        #
        # MaxPool1d:
        # 476 -> 36
        #
        # BiLSTM: 320 forward + 320 backward = 640
        # Self-attention (if enabled): stays at 320, no directional doubling
        #
        # Attention pooling collapses the CNN time steps to one vector
        # (weighted by learned per-position importance) instead of
        # flattening all positions into a fixed-size vector - same
        # final output width (925), far fewer parameters in this layer.
        self.attention_pool = AttentionPooling(
            input_dim=sequence_encoder_output_dim,
            attention_hidden_dim=128,
        )

        self.last_attention_weights: torch.Tensor | None = None

        self.fc1 = nn.Linear(
            sequence_encoder_output_dim,
            925,
        )

    def forward(
        self,
        seq_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            seq_input:
                One-hot DNA sequences with shape
                [B, 4, 501].

        Returns:
            Sequence representations with shape
            [B, 925].
        """

        if self.use_multiscale_cnn:
            # Each branch: [B, 4, 501] -> [B, MULTISCALE_CHANNELS_PER_BRANCH, 501]
            # ('same' padding keeps every branch's length equal regardless
            # of its own kernel size, so they can be concatenated).
            branch_outputs = [
                F.relu(conv(seq_input)) for conv in self.multiscale_convs
            ]
            # -> [B, TOTAL_CNN_CHANNELS, 501]
            seq_output = torch.cat(branch_outputs, dim=1)
        else:
            # [B, 4, 501] -> [B, 320, 476]
            seq_output = self.conv1(
                seq_input
            )

            seq_output = F.relu(
                seq_output
            )

        # [B, 320, 476] -> [B, 320, 36] (single-branch) or
        # [B, 320, 501] -> [B, 320, 38] (multiscale)
        seq_output = self.maxpool(
            seq_output
        )

        seq_output = self.drop1(
            seq_output
        )

        # Conv1d produces [B, channels, length].
        # batch_first=True LSTM/Transformer expects [B, length, features].
        #
        # [B, 320, L] -> [B, L, 320]
        seq_output = seq_output.permute(
            0,
            2,
            1,
        )

        if self.use_self_attention:
            # [B, L, 320] + positional embedding -> self-attention -> [B, L, 320]
            seq_output = seq_output + self.positional_embedding.unsqueeze(0)
            seq_output = self.self_attention_encoder(seq_output)
        else:
            # [B, L, 320] -> [B, L, 640]
            seq_output, _ = self.bilstm(
                seq_output
            )

        seq_output = self.drop2(
            seq_output
        )

        # [B, L, D] -> [B, D] (attention-weighted, not flattened)
        seq_output, attention_weights = self.attention_pool(
            seq_output
        )

        # Stored for interpretability (e.g. plotting which motif regions
        # the model attended to) - not read anywhere in the forward
        # pass itself, so it never affects gradients/training.
        self.last_attention_weights = attention_weights.detach()

        # [B, 640] -> [B, 925]
        seq_output = self.fc1(
            seq_output
        )

        seq_output = F.relu(
            seq_output
        )

        return seq_output
