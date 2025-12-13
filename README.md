(douaa) \
So far the steps you have to follow to be on the same page as me is:
1. Create a virtual envirenement with `python -m venv .venv`
2. Activate the virtual environement with `source .venv/Scripts/activate` (I guess this is on windows idk for linux)
3. Run `pip install -r requirements.txt`
4. Install tesseract-OCR from https://github.com/tesseract-ocr/tesseract (i think?)
5. Download the french data file from https://github.com/tesseract-ocr/tessdata/blob/main/fra.traineddata
6. Add the french data file to "C:\Program Files\Tesseract-OCR\tessdata\" (**Optional:** You can check if it worked by running `tesseract --list-langs` in the cmd. You should see **fra** if it did :) )

You guys can get rid of the .gitkeep files when you put a file in the folder, i just used them so the structure is visible on github (aka when u pull the changes)

Pretty obvious but if you want any files or folders to be ignored put them in .gitignore and if you download any new libraries run `pip freeze > requirement.txt` from the root folder (do NOT do it before you install the requirements otherwise you'll lose them and don't use >> or you'll have doubles )