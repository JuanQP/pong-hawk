# Pong Hawk

👷‍♂️ This project is still under development 👷‍♀️

![pong-hawk-debug](https://github.com/JuanQP/pong-hawk/assets/11776905/e80f22fa-535a-4f0e-8b23-78a9b16ab327)

![pong-hawk-heatmap](https://github.com/JuanQP/pong-hawk/assets/11776905/efccdac2-0231-4d2e-a51b-7f9982a55488)

# Development

You need the `trained_model.pt` file in the `model` folder and `poetry` installed in order to execute this project. Then:

```sh
poetry install
```

# Running the project

If you want to **run the model with images**, you have to put your images in the folder `images`, then, depending on what you want to do, run the project with either:

```sh
poetry run python images.py
poetry run python videos.py
```

Depending on which command you used, it will process the `images` folder or the `videos` folder and then export the results with the `pong-hawk-` suffix in the file name.

In case of processing images, it will process every single image in that directory, but in the case of videos a prompt will appear asking you which file you want to process.
