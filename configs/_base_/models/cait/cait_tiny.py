# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='Cait3D',
        depth=3,
        cls_depth=2,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
        layer_dropout=0.0))