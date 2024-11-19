import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from utilities import read_json_from_folder, convertStringListToString
from utilities import stop_words


# Step 1: Read and preprocess your data
folder_path = 'ECHR_Dataset/EN_train'
folder_path2 = 'ECHR_Dataset/EN_test'

# Assuming a function that reads your JSON data into a DataFrame
train_data = read_json_from_folder(folder_path)
test_data = read_json_from_folder(folder_path2)

# Combine the data from both folders into one list
combined_data = train_data + test_data
df = pd.DataFrame(combined_data)
print("Completed framing data into df")

text = ' '.join(df['TEXT'].apply(convertStringListToString))

wordcloud = WordCloud(stopwords=stop_words, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()