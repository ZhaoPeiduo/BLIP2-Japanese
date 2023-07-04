# BLIP2-Japanese

This project builds upon [LAVIS](https://github.com/salesforce/LAVIS) library's BLIP2 mdoel.

The main idea is to replace the tokenizer and the underlying BERT model in Blip2's Qformer with the one trained on Japanese datasets and retrain the upated model on Japanese captioning datasets.

The model has been trained for stage1 using COCO dataset with [STAIR captions](http://captions.stair.center/#:~:text=STAIR%20Captions%20is%20a%20large,multimodal%20retrieval%2C%20and%20image%20generation.).

## Use Case: Generate Japanese Captions for Captioning Datasets

The weights of Blip2_Japanese_qformer trained on STAIR can be obtained from [this link](https://drive.google.com/drive/folders/11YRyQb-_Pn8g3Wlnv2aBwNnvZ0Oo4LRM?usp=drive_link).

Copy the whole folder under lavis directory to run the example jupyter notebook.

Captions generated for [flickr30k dataset](https://www.kaggle.com/datasets/adityajn105/flickr30k?select=Images) can be found in flickr30k_caption.json. Script in flickr30k_caption_generate.ipynb. 

These captions are generated using top-k sampling instead of nucleus, and may contain wrong details as shown in the examples below:

![1001773457](https://github.com/ZhaoPeiduo/BLIP2-Japanese/assets/77187494/eae2e401-9697-45ad-b118-4c8ea7ae95f4)

 {'image': '1001773457.jpg', 'caption': ['二 匹 の 犬 が 道路 で フリスビー を し て いる']} # No frisbee

 ![1001573224](https://github.com/ZhaoPeiduo/BLIP2-Japanese/assets/77187494/9a563146-e815-49e7-96d4-55a69a3d0123)
 
{'image': '1001573224.jpg', 'caption': ['6 人 の 女性 が 屋内 で 飛び跳ね て いる']} # Wrong head count

## Use Case: Image Retrieval

Refer to the example.ipynb notebooks for more details. The idea is to get the average cosine similarity of query tokens between the image embeddings and the multimodal embeddings.

This model is still experimental and might be further trained on other captioning datasets with Japanese captions.
