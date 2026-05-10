from PIL import Image, ImageDraw
from pathlib import Path

d = Path("data/test_images")
d.mkdir(parents=True, exist_ok=True)

img = Image.new("RGB", (640, 480), color=(135, 206, 235))
draw = ImageDraw.Draw(img)
draw.rectangle([0, 320, 640, 480], fill=(80, 80, 80))
draw.rectangle([80, 240, 280, 340], fill=(200, 50, 50))
draw.rectangle([100, 220, 260, 250], fill=(200, 50, 50))
draw.ellipse([420, 190, 460, 230], fill=(255, 220, 180))
draw.rectangle([415, 230, 465, 320], fill=(50, 50, 200))
draw.rectangle([540, 260, 560, 330], fill=(101, 67, 33))
draw.ellipse([510, 200, 590, 270], fill=(34, 139, 34))
img.save(d / "street.jpg", "JPEG")
img.save(d / "office.jpg", "JPEG")
print("✅ 테스트 이미지 생성 완료")
print(f"저장 위치: {d.resolve()}")
