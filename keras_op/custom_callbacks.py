from tensorflow.keras import callbacks

class StopTraining(callbacks.Callback):
    def __init__(self, monitor='val_loss', patience=10):
        self.monitor = monitor
        self.patience = patience

    def on_epoch_end(self, epoch, logs={}):
        current_val_acc = logs.get(self.monitor)
        
        if current_val_acc < 0.5 and epoch == self.patience:
            self.model.stop_training = True
