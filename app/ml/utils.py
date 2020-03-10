from uuid import uuid4
import matplotlib.pylab as plt
from wordcloud import WordCloud
from many_stop_words import get_stop_words

from app.core.config import STATIC_DIR

def build_wordcloud(text, lang):
    stop_words = get_stop_words(lang)
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stop_words,
        max_words=200,
        max_font_size=40,
        random_state=42
    ).generate(str(text))    

    target_dir = "wordcloud"
    filename = f"{uuid4().hex[:10]}.png"

    fig = plt.figure(1)
    plt.imshow(wordcloud)
    plt.axis("off")
    out_file = f"{target_dir}/{filename}"
    plt.savefig(STATIC_DIR.joinpath(out_file), bbox_inches="tight")    

    return f"http://localhost:8000/static/{out_file}"
    