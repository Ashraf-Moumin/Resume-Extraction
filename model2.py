from transformers import pipeline

model = pipeline("summarization", model= "Ameer05/distilbart-cnn-12-6-finetuned-resume-summarizer")




def summarize(text):
    summarized_text = ""
    #since the length of the tokinized input is not the same as splitting
    #by spaces, we approximate
    length = len(5*text.split())
    
    for i in range(length//1000):
        if 3*len(text[i*1000:(i+1)*1000].split()) > 35:
            summarized_text+= " " + model(text[i*1000:(i+1)*1000])[0]["summary_text"]
    
    return summarized_text
    
