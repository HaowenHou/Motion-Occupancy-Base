import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8, batch_first=True)
        self.ln1 = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim*4),
            nn.ELU(),
            nn.Linear(feature_dim*4, feature_dim),
        )
        self.ln2 = nn.LayerNorm(feature_dim)

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x)[0])
        x = self.ln2(x + self.mlp(x))
        return x
        
class MotionTransformer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        IN_DIM = config.MODEL.IN_DIM
        OUT_DIM = config.MODEL.OUT_DIM
        mconfig = config.MODEL.BASE_MOTION_TRANSF
        FEATURE_DIM = mconfig.EXPERT_HIDDEN_SIZE
        QUERY_NUM = mconfig.QUERY_NUM
        DROPOUT = mconfig.TRANSF_DROPOUT

        self.input_proj = nn.Linear(IN_DIM, FEATURE_DIM)
        self.attn1 = nn.MultiheadAttention(embed_dim=FEATURE_DIM, num_heads=8, dropout=DROPOUT, batch_first=True)
        self.query1 = nn.Parameter(torch.randn(QUERY_NUM, FEATURE_DIM))
        self.ln1_1 = nn.LayerNorm(FEATURE_DIM)
        self.mlp1 = nn.Sequential(
            nn.Linear(FEATURE_DIM, FEATURE_DIM*4),
            nn.ELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FEATURE_DIM*4, FEATURE_DIM),
            nn.Dropout(DROPOUT),
        )
        self.ln1_2 = nn.LayerNorm(FEATURE_DIM)

        self.transformer_blocks = nn.Sequential(TransformerBlock(FEATURE_DIM))

        self.attn2 = nn.MultiheadAttention(embed_dim=FEATURE_DIM, num_heads=8, dropout=DROPOUT, batch_first=True)
        self.query2 = nn.Parameter(torch.randn(1, FEATURE_DIM))
        self.ln2_1 = nn.LayerNorm(FEATURE_DIM)
        self.mlp2 = nn.Sequential(
            nn.Linear(FEATURE_DIM, FEATURE_DIM*4),
            nn.ELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FEATURE_DIM*4, FEATURE_DIM),
            nn.Dropout(DROPOUT),
        )
        self.ln2_2 = nn.LayerNorm(FEATURE_DIM)

        self.output_proj = nn.Linear(FEATURE_DIM, OUT_DIM)
        

    def forward(self, x):
        bs = x.size(0)
        x = self.input_proj(x) # [N, 512]
        x = x[:, None, :]
        x = self.attn1(self.query1[None].repeat(bs, 1, 1), x, x)[0] # [N, 5, 512]
        x = self.ln1_1(x)
        x = self.ln1_2(x + self.mlp1(x))

        x = self.transformer_blocks(x)

        x = self.attn2(self.query2.repeat(bs, 1, 1), x, x)[0] # [N, 1, 512]
        x = self.ln2_1(x)
        x = self.ln2_2(x + self.mlp2(x))
        x = self.output_proj(x)
        return x.squeeze(1)

class BaseMotionTransf(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.CONFIG = config
        mconfig = config.MODEL.BASE_MOTION_TRANSF
        DROPOUT = mconfig.GATING_DROPOUT

        self.EXPERT_NUM = mconfig.EXPERT_NUM
        GATING_HIDDEN_SIZE = mconfig.GATING_HIDDEN_SIZE
        self.expert_mlp = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(config.MODEL.IN_DIM, GATING_HIDDEN_SIZE),
            nn.ELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(GATING_HIDDEN_SIZE, GATING_HIDDEN_SIZE),
            nn.ELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(GATING_HIDDEN_SIZE, self.EXPERT_NUM)
        )

        self.experts = nn.ModuleList()
        for _ in range(self.EXPERT_NUM):
            self.experts.append(MotionTransformer(config))

    def forward(self, data_input):
        x = data_input

        assert not torch.isnan(x).any()
        if self.EXPERT_NUM == 1:
            return self.experts[0](x)
        
        expert_weights = self.expert_mlp(x)
        expert_weights = nn.functional.softmax(expert_weights, dim=-1)

        final_output = []
        for i in range(self.EXPERT_NUM):
            expert_output = self.experts[i](x)
            final_output.append(expert_weights[:, i:i+1] * expert_output)
        final_output = torch.stack(final_output).sum(0)

        return final_output
    