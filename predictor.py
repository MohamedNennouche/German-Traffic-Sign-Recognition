import tensorflow as tf
import numpy as np

def predict_with_model(model, img_path) : 
    image = tf.io.read_file(img_path) # On lit l'image
    image = tf.image.decode_png(image, channels=3) # On la décode
    image = tf.image.convert_image_dtype(image, dtype=tf.float32) # Convertir les entiers en float, cela permet de rescaler nos images
    image = tf.image.resize(image , [60,60]) # resize les images et on a alors de la forme (60,60,3)
    image = tf.expand_dims(image, axis=0) # pour avoir au final (1,60,60,3) pour l'adapter à notre modèle (voir summary du modèle il attend un tel format)

    prediction = model.predict(image) # peut etre une décision ou un ensemble de probabilités (pour chaque classe)
    prediction = np.argmax(prediction) # Pour avoir l'indexe de la meilleure probabilité et par conséquent le label
    return prediction

if __name__=='__main__' :

    img_path = "./Test/10/00027.png"
    model = tf.keras.models.load_model('./Models')
    prediction = predict_with_model(model, img_path)
    print(f'prediction : {prediction}')