from paddleocr import PaddleOCR
from pathlib import Path

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

folder = Path("good_test_images") #change folder when nec

for image_path in folder.glob("*.png"):

    print("\n")
    print("=" * 60)
    print(f"IMAGE: {image_path.name}")
    print("=" * 60)

    results = ocr.predict(str(image_path))

    for result in results:

        data = result.json["res"]

        texts = data["rec_texts"]
        scores = data["rec_scores"]

        for text, score in zip(texts, scores):
            print(f"{score:.3f} | {text}")