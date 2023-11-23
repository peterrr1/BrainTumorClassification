
import tensorflow as tf
import pathlib
import os
import matplotlib.pyplot as plt
import shutil
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32
IMG_WIDTH = 224
IMG_HEIGHT = 224
EPOCHS = 20


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    label_ = parts[-2] == class_names
    return tf.argmax(label_)


def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
    label_ = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label_


def configure_for_performance(ds):
    #ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


# creates a subset of the original dataset
def create_optimization_directory():
    try:
        os.mkdir("optiData")
        [os.mkdir(f"./optiData/{dir_}") for dir_ in os.listdir('./Data') if dir_ != '.DS_Store']
        for dir_ in os.listdir('./optiData'):
            num_of_files = len(os.listdir(f'./optiData/{dir_}'))
            print(f'Num of files in {dir_} ' + str(num_of_files))
            for file in os.listdir(f'./Data/{dir_}'):
                if num_of_files == 100:
                    break
                shutil.copy(f'./Data/{dir_}/{file}', f'./optiData/{dir_}')
                num_of_files += 1
            print(f'Num of files in {dir_} ' + str(num_of_files))
    except FileExistsError:
        print('File or directory already exists')


if __name__ == '__main__':

    #create_optimization_directory()
    data_lib = pathlib.Path('./Data')
    list_ds = tf.data.Dataset.list_files(str(data_lib/'*/*'), shuffle=False)
    image_count = len(list_ds)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
    class_names = np.array(sorted([item.name for item in data_lib.glob('*') if item.name != '.DS_Store']))
    print(class_names)
    val_size = int(image_count * 0.2)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)

    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(filters=48, kernel_size=(11, 11), strides=4, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=2, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Conv2D(filters=196, kernel_size=(3, 3), strides=1, activation='relu'),
        tf.keras.layers.Conv2D(filters=196, kernel_size=(3, 3), strides=1, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax'),

    ])
    """
    outp = model.predict(train_ds)
    print(outp.shape)
    """

    model.compile(
        optimizer='rmsprop',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    analytics = model.fit(
        train_ds,
        validation_data=val_ds,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )

    plt.plot(analytics.history['accuracy'], label='Accuracy', marker='o', linestyle='--', color='r')
    plt.plot(analytics.history['val_accuracy'], label='Validation accuracy', marker='o', linestyle='--', color='g')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.xticks(np.arange(1, EPOCHS, step=1))
    plt.legend(loc='lower right')
    plt.show()
    test_loss, test_acc = model.evaluate(val_ds, verbose=2)
    model.summary()
    model.save('saved_model/my_model_3')
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalFocalCrossentropy(),
        metrics=['accuracy']
    )

    analytics = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    plt.plot(analytics.history['accuracy'], label='Accuracy', marker='o', linestyle='--', color='r')
    plt.plot(analytics.history['val_accuracy'], label='Validation accuracy', marker='o', linestyle='--', color='g')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.xticks(np.arange(1, EPOCHS, step=1))
    plt.legend(loc='lower right')
    plt.show()
    test_loss, test_acc = model.evaluate(val_ds, verbose=2)
    model.summary()

    
    data_lib = pathlib.Path('Data/')
    list_ds = tf.data.Dataset.list_files(str(data_lib/'*/*'), shuffle=False)
    image_count = len(list_ds)
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

    class_names = np.array(sorted(([item.name for item in data_lib.glob('*') if item.name != '.DS_Store'])))

    val_size = int(image_count * 0.2)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)

    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)

    image_batch, label_batch = next(iter(train_ds))

    image = image_batch[0].numpy()
    image = image.reshape(1, IMG_WIDTH, IMG_HEIGHT, 3)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation='relu'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(2, 2), strides=1, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=1, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=1, activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=1, activation='relu'),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, activation='relu'),
        #tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(2, 2), strides=2, activation='relu'),
        tf.keras.layers.Conv2D(filters=1024, kernel_size=(3, 3), strides=1, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(4)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.xlim([1, 7])
    plt.legend(loc='lower right')
    plt.show()
    test_loss, test_acc = model.evaluate(val_ds, verbose=2)
    model.summary()
    model.save('saved_model/my_model_3')
    """
