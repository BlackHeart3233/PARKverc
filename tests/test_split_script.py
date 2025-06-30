import os
import tempfile
import pytest
from model_1.split_script import split_dataset

@pytest.fixture
def setup_dummy_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = os.path.join(tmpdir, "images")
        lbl_dir = os.path.join(tmpdir, "labels")
        out_dir = os.path.join(tmpdir, "output")

        os.makedirs(img_dir)
        os.makedirs(lbl_dir)

        for i in range(10):
            img_name = f"img_{i}.jpg"
            with open(os.path.join(img_dir, img_name), "w") as f:
                f.write("image data")

            if i != 9:
                with open(os.path.join(lbl_dir, f"img_{i}.txt"), "w") as f:
                    f.write("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2")

        yield img_dir, lbl_dir, out_dir

def test_split_dataset_creates_folders(setup_dummy_data):
    img_dir, lbl_dir, out_dir = setup_dummy_data
    split_dataset(img_dir, lbl_dir, out_dir, split_ratio=0.8)

    for split in ['train', 'val']:
        for sub in ['images', 'labels']:
            path = os.path.join(out_dir, sub, split)
            assert os.path.exists(path)
            assert len(os.listdir(path)) > 0

def test_split_dataset_respects_label_matching(setup_dummy_data, capfd):
    img_dir, lbl_dir, out_dir = setup_dummy_data
    split_dataset(img_dir, lbl_dir, out_dir, split_ratio=0.5)

    out = capfd.readouterr().out
    assert "Label datoteka" in out
