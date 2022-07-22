from flask import Flask,render_template,request,url_for
import numpy as np
import tensorflow as tf
from transformers import TFBertModel
from transformers import AutoTokenizer

from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# model = BertModel.from_pretrained("bert-base-cased")
model = tf.keras.models.load_model('/home/ravikumar/Desktop/FeedBack_Prize/Model.h5')


def bert_encoder(texts,tokenizer,max_len = 256):
    input_ids  = list()
    token_type_ids = list()
    attention_mask = list()
    
    for text in texts:
        token  = tokenizer(text,max_length = 256,truncation = True,padding = 'max_length',add_special_tokens = True)
        input_ids.append(token['input_ids'])
        token_type_ids.append(token['token_type_ids'])
        attention_mask.append(token['attention_mask'])
        
    return np.array(input_ids),np.array(token_type_ids),np.array(attention_mask)
app = Flask(__name__)



@app.route("/",methods = ['GET'])
def hello_word():
    return render_template('home.html')

@app.route("/",methods = ['POST'])
def predict():
    if request.method == 'POST':
        language = request.form.get('discourse_text')
        framework = request.form.get('discourse_type')
        out = language + "[SEP]" + framework
        test_text = bert_encoder([str(out)], tokenizer)
        preds = model.predict(test_text, verbose=1)
        Ineffective = preds[:,0] > 0.5
        Adequate = preds[:,1] > 0.5
        Effective = preds[:,2] >0.5

        return '''
                  <h1 style="text-align:center" >Feedback-Effective-Argument-Prediction</h1>
                  <br>
                  <h1 style="text-align:center" >Ineffective : {}</h1>
                  <h1 style="text-align:center" >Adequate : {}</h1>
                  <h1 style="text-align:center" >Effective : {}</h1>'''.format(Ineffective,Adequate,Effective)


if  __name__ == '__main__':
    app.run(port = 5000,debug=True) 