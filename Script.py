import streamlit as st
import numpy as np
from PIL import Image 
from tensorflow.keras.models import load_model
import tensorflow as tf
 
from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image 

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

def loading_model():
  fp = "cnn_pneu_vamp_model.h5"
  model_loader = load_model(fp)
  return model_loader

cnn = loading_model()
st.write("""
# Respiratory Disease Classification using X-ray
-
""")
 


temp = st.file_uploader("""Upload X-Ray Image""")
#temp = temp.decode()

buffer = temp
temp_file = NamedTemporaryFile(delete=False)
if buffer:
    temp_file.write(buffer.getvalue())
    st.write(image.load_img(temp_file.name))

if buffer is None:
  st.text("Try again.")

else:
  temp_img = image.load_img(temp_file.name, target_size=(500, 500),color_mode='grayscale')

  # Preprocessing the image
  pp_temp_img = image.img_to_array(temp_img)
  pp_temp_img = pp_temp_img/255
  pp_temp_img = np.expand_dims(pp_temp_img, axis=0)

  #predict
  preds = cnn.predict(pp_temp_img)
  if preds>= 0.5:
    out = ('I am {:.2%} percent confirmed that this is a Pneumonia case'.format(preds[0][0]))
  
  else: 
    out = ('I am {:.2%} percent confirmed that this is a Normal case'.format(1-preds[0][0]))
  st.success(out)
  image = Image.open(temp)
  st.image(image,use_column_width=True)