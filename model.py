
import tensorflow as tf

import numpy as np

text = open('parsedMessages.txt', 'r').read()

vocab = sorted(set(text))
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

intText = np.array([char2idx[c] for c in text])


def buildModel(nx, ne, nu, batchSize):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(nx, ne, batch_input_shape=[batchSize, None]))
    model.add(tf.keras.layers.LSTM(nu, return_sequences=True))
    model.add(tf.keras.layers.Dropout(.5))
    model.add(tf.keras.layers.LSTM(nu, return_sequences=True))
    model.add(tf.keras.layers.Dense(nx))
    return model


def gettarg(chunk):
    return chunk[:-1], chunk[1:]


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def preprocess():

    seqLen = 32
    charDataset = tf.data.Dataset.from_tensor_slices(intText)
    seq = charDataset.batch(seqLen + 1, drop_remainder=True)

    dataset = seq.map(gettarg)

    BATCH_SIZE = 64
    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    model = buildModel(len(vocab), 256, 512, BATCH_SIZE)
    model.load_weights(tf.train.latest_checkpoint('./checkpts'))
    print(model.summary())

    model.compile(optimizer='adam', loss=loss)

    checkpointCallback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./checkpts/ckpt{epoch}',
        save_weights_only=True)

    EPOCHS = 2

    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpointCallback])
    return history


def predict(start):
    model = buildModel(len(vocab), 256, 512, 1)
    model.load_weights('./checkpts/')
    model.build(tf.TensorShape([1, None]))
    print(model.summary())

    Ty = 32
    inp = [char2idx[s] for s in start]
    inp = tf.expand_dims(inp, 0)

    gen = []
    temp = .8

    model.reset_states()
    for i in range(Ty):
        pred = model(inp)
        pred = tf.squeeze(pred, 0)
        pred = pred / temp
        predID = tf.random.categorical(pred, num_samples=1)[-1, 0].numpy()

        inp = tf.expand_dims([predID], 0)
        gen.append(idx2char[predID])

    return (start + ''.join(gen))


preprocess()
