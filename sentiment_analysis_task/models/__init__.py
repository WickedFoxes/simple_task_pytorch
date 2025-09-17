from .lstm_classfication import LSTMClassifier

def build_model(model_name : str,
            vocab_size,
            num_classes,
            embed_dim,
            hidden_dim,
            num_layers,
            bidirectional,
            dropout,
            pad_idx
    ):
    if model_name == 'lstm_classification':
        model = LSTMClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            pad_idx=pad_idx
        )
    return model
    