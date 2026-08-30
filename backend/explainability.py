def get_feature_importance(model):
    try:
        importance = model.feature_importances_
        return importance.tolist()
    except:
        return []
