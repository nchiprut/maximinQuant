# %%
import tensorflow as tf
import numpy as np
import larq as lq

# %%
class QuantModel(tf.keras.Model):
    def __init__(self, seq_layers, quant_var):
        super().__init__(name='')
        self.seq_layers = seq_layers
        self.quant_var = quant_var

    def compile(self, *args, loss_fn=None, metric_clss=None, max_opt=None, lamda=1.,
                quant_forward=True, **kwargs):
        super().compile(*args, **kwargs)
        self.quant_forward = quant_forward
        self.lamda = lamda
        self.r_metrics = [met('r_' + met.__name__) for met in metric_clss]
        self.q_metrics = [met('q_' + met.__name__) for met in metric_clss]

        self.r_loss = tf.keras.metrics.Mean(name="r_loss")
        self.q_loss = tf.keras.metrics.Mean(name="q_loss")
        self.loss_fn = loss_fn

        self.max_opt = max_opt
        # self.flat_w = None
        self(np.zeros(shape=(1,) + self.seq_layers[0].input_shape[0][1:], dtype=np.float32))
        self.z = None if self.max_opt is None else \
             tf.Variable(tf.zeros(
                 shape=(tf.reduce_sum([tf.reduce_prod(var.shape) for var in self.trainable_variables if 'batch_normalization' not in var.name]), 1)
                 ), trainable=False, name='z')
    
    def call(self, input_tensor, training=False):
        x = input_tensor
        for layer in self.seq_layers:
            x = layer(x, training=training)
        return x

    def train_step(self, data):
        x, y = data

        self.quant_var.assign(not self.quant_forward)
        n_y_pred = self(x, training=False)
        n_loss = self.loss_fn(y, n_y_pred)

        self.quant_var.assign(self.quant_forward)
        with tf.GradientTape(True) as w_tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.loss_fn(y, y_pred)

            flat_w = tf.concat(
                [tf.reshape(Wb, (-1, 1)) for Wb in self.trainable_variables \
                        if 'batch_normalization' not in Wb.name],
                        axis=0)

            min_loss = loss + self.lamda * tf.reduce_mean(self.z * (1. - flat_w**2))

        w_gradients = w_tape.gradient(min_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(w_gradients, self.trainable_variables))

        if self.max_opt is not None:
            with tf.GradientTape(True) as z_tape:
                z_tape.watch(self.z)
                y_pred = self(x, training=True)  # Forward pass
                loss = self.loss_fn(y, y_pred)

                flat_w = tf.concat(
                    [tf.reshape(Wb, (-1, 1)) for Wb in self.trainable_variables \
                            if 'batch_normalization' not in Wb.name],
                            axis=0)
                max_loss = -loss - self.lamda * tf.reduce_mean(self.z * (1. - flat_w**2))

            z_gradients = z_tape.gradient(max_loss, self.z)
            self.max_opt.apply_gradients([(z_gradients, self.z)])

        if self.quant_forward:
            y_pred, n_y_pred = n_y_pred, y_pred
            loss, n_loss = n_loss, loss

        [ m.update_state(y, y_pred) for m in self.r_metrics ]
        [ m.update_state(y, n_y_pred) for m in self.q_metrics ]
        self.r_loss.update_state(loss)
        self.q_loss.update_state(n_loss)
        ret = {
            **{m.name: m.result() for m in self.metrics},
            **{m.name: m.result() for m in self.q_metrics},
            **{m.name: m.result() for m in self.r_metrics},
        }
        [ m.reset_states() for m in self.r_metrics ]
        [ m.reset_states() for m in self.q_metrics ]
        self.r_loss.reset_states()
        self.q_loss.reset_states()
        return ret

    def test_step(self, data):
        x, y = data
        self.quant_var.assign(False)
        r_y_pred = self(x, training=False)

        self.quant_var.assign(True)
        q_y_pred = self(x, training=False)

        q_loss = self.loss_fn(y, q_y_pred)
        r_loss = self.loss_fn(y, r_y_pred)
        [ m.update_state(y, r_y_pred) for m in self.r_metrics ]
        [ m.update_state(y, q_y_pred) for m in self.q_metrics ]
        self.r_loss.update_state(r_loss)
        self.q_loss.update_state(q_loss)
        return {
            **{m.name: m.result() for m in self.metrics},
            **{m.name: m.result() for m in self.q_metrics},
            **{m.name: m.result() for m in self.r_metrics},
        }
