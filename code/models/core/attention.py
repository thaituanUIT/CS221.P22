import tensorflow as tf 

def attention(inputs):
    # Trainable parameters
    hidden_size = inputs.shape[2].value
    u_omega = tf.get_variable("u_omega", [hidden_size], initializer=tf.keras.initializers.glorot_normal())

    with tf.name_scope('v'):
        v = tf.tanh(inputs)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    # Final output with tanh
    output = tf.tanh(output)

    return output, alphas

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, hidden_size, **kwargs):
        """
        Multi-head attention layer.
        """
        super(MultiHeadAttention, self).__init__(**kwargs)
        if num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.depth = hidden_size // num_heads

        # Define projection layers
        self.W_q = tf.keras.layers.Dense(hidden_size)
        self.W_k = tf.keras.layers.Dense(hidden_size)
        self.W_v = tf.keras.layers.Dense(hidden_size)

        self.final_linear = tf.keras.layers.Dense(hidden_size)

    def split_heads(self, x, batch_size):
        # x: (batch_size, seq_len, hidden_size)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))  # (B, T, H, D)
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (B, H, T, D)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # q, k, v: (B, H, T, D)
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (B, H, T, T)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (B, H, T, T)
        output = tf.matmul(attention_weights, v)  # (B, H, T, D)

        return output, attention_weights

    def call(self, x, mask=None):
        batch_size = tf.shape(x)[0]

        q = self.W_q(x)  # (B, T, hidden_size)
        k = self.W_k(x)
        v = self.W_v(x)

        q = self.split_heads(q, batch_size)  # (B, H, T, D)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)  # (B, H, T, D)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (B, T, H, D)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.hidden_size))  # (B, T, hidden_size)

        output = self.final_linear(concat_attention)  # (B, T, hidden_size)

        return output, attention_weights
