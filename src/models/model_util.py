
def load_model(cfg):
    model = None
    if cfg.task == "classification":
        from src.models.pose_classifier import PoseClassifier
        model = PoseClassifier(input_shape=(cfg.train.batch_size, cfg.in_channels, cfg.depth, cfg.width, cfg.height), output_dim=cfg.output_dim)
    if cfg.task == "privacy":
        from src.models.privacy_classifier import PrivacyClassifier
        model = PrivacyClassifier(input_shape=(cfg.train.batch_size, cfg.in_channels, cfg.depth, cfg.width, cfg.height), output_dim=(cfg.privacy_attributes.skin_color, cfg.privacy_attributes.gender, cfg.privacy_attributes.age))
    return model