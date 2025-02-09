import deepl
import pandas as pd


def safe_translate(deepl_cl, text):
    try:
        translation = deepl_cl.translate_text(text, target_lang="EN-US")
        print(f"{text}\n->\n{translation}\n")
        return translation
    except Exception as e:
        print(f"Error while translating: {e}\n")
        return text


if __name__ == "__main__":
    df = pd.read_csv("../track_a/train/deu.csv")
    auth_key = "****"   # needs to be entered
    deepl_client = deepl.DeepLClient(auth_key)

    # remove disgust class as it is not present in the english dataset
    df = df[df["disgust"] != 1].drop(columns=["disgust"])

    df["text"] = df["text"].apply(lambda x: safe_translate(deepl_client, x))

    print(df.to_string(index=False))

    df.to_csv("../track_a/train/deepl_deu-to-eng.csv", index=False)
