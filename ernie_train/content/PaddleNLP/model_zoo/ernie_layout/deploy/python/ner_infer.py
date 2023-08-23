from paddleocr import PaddleOCR
from ner_predictor import Predictor

class NERArgs():
    model_path_prefix = r'D:\Dev\tabular_doc_analysis\ernie_train\content\PaddleNLP\model_zoo\ernie_layout\export\inference'
    task_type = 'ner'
    lang = 'en'
    batch_size = 8
    max_seq_length = 512
    device = 'gpu'

def main():

    
    docs = [r"D:\Dev\tabular_doc_analysis\kaggle2.png"]
 
    args = NERArgs()
    predictor = Predictor(args)

    outputs = predictor.predict(docs)
    import pprint
    from PIL import Image, ImageDraw, ImageFont
    import json
    
    image = Image.open(docs[0]).convert("RGB")
    
    draw = ImageDraw.Draw(image)

    font = ImageFont.load_default()

    label2color = {'question':'blue', 'answer':'green', 'header':'orange', 'other':'violet'}

    for output in outputs[0]['result']:
        box = output['bbox']
        predicted_label = output['label'].lower()
        draw.rectangle(box, outline=label2color[predicted_label])
        draw.text((box[0] + 10, box[1] - 10), text=predicted_label, fill=label2color[predicted_label], font=font)

    with open('output4.json', 'w') as fp:
        json.dump(outputs, fp)
    image.save('output4.png')

    pprint.sorted = lambda x, key=None: x
    pprint.pprint(outputs)
    

if __name__ == "__main__":
    main()
