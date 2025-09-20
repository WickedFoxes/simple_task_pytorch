from sklearn.metrics import classification_report, confusion_matrix, f1_score

def compute_f1(y_true, y_pred, average="macro"):
    return f1_score(y_true, y_pred, average=average)

def compute_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def compute_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred, output_dict=True)