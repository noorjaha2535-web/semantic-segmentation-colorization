{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 715
        },
        "id": "Lbaw5vqWk4qp",
        "outputId": "2f13d57e-0bf9-4996-c42c-67d313cdcfea"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.12/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1`. You can also use `weights=DeepLabV3_ResNet101_Weights.DEFAULT` to get the most up-to-date weights.\n",
            "  warnings.warn(msg)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Downloading: \"https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth\" to /root/.cache/torch/hub/checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "100%|██████████| 233M/233M [00:01<00:00, 162MB/s]\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Colab notebook detected. To show errors in colab notebook, set debug=True in launch()\n",
            "* Running on public URL: https://87357a3c9b3dfb000f.gradio.live\n",
            "\n",
            "This share link expires in 1 week. For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "<div><iframe src=\"https://87357a3c9b3dfb000f.gradio.live\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": []
          },
          "metadata": {},
          "execution_count": 1
        }
      ],
      "source": [
        "# =========================================================\n",
        "# TASK 5: Semantic Segmentation for Targeted Colorization\n",
        "# Author : Mehkaan Anjum\n",
        "# Company: Elevance Skills\n",
        "# Description:\n",
        "# Selectively colorizes foreground or background\n",
        "# using semantic segmentation with a GUI.\n",
        "# =========================================================\n",
        "\n",
        "import torch\n",
        "import torchvision.transforms as T\n",
        "from torchvision import models\n",
        "import numpy as np\n",
        "import gradio as gr\n",
        "from PIL import Image\n",
        "\n",
        "# Load pretrained DeepLabV3 segmentation model\n",
        "model = models.segmentation.deeplabv3_resnet101(pretrained=True)\n",
        "model.eval()\n",
        "\n",
        "# Image preprocessing\n",
        "transform = T.Compose([\n",
        "    T.Resize((256, 256)),\n",
        "    T.ToTensor(),\n",
        "    T.Normalize(mean=[0.485, 0.456, 0.406],\n",
        "                std=[0.229, 0.224, 0.225])\n",
        "])\n",
        "\n",
        "def segment_and_colorize(image, region_choice):\n",
        "    image = image.convert(\"RGB\")\n",
        "    input_tensor = transform(image).unsqueeze(0)\n",
        "\n",
        "    with torch.no_grad():\n",
        "        output = model(input_tensor)[\"out\"][0]\n",
        "\n",
        "    segmentation = output.argmax(0).numpy()\n",
        "    image_np = np.array(image.resize((256, 256)))\n",
        "\n",
        "    # COCO class 15 = Person (foreground)\n",
        "    mask = segmentation == 15\n",
        "\n",
        "    # Apply simple color overlay (blue tint)\n",
        "    color_layer = image_np.copy()\n",
        "    color_layer[:, :, 2] = np.clip(color_layer[:, :, 2] + 100, 0, 255)\n",
        "\n",
        "    if region_choice == \"Foreground\":\n",
        "        image_np[mask] = color_layer[mask]\n",
        "    else:\n",
        "        image_np[~mask] = color_layer[~mask]\n",
        "\n",
        "    return Image.fromarray(image_np)\n",
        "\n",
        "# Gradio GUI\n",
        "interface = gr.Interface(\n",
        "    fn=segment_and_colorize,\n",
        "    inputs=[\n",
        "        gr.Image(type=\"pil\", label=\"Upload Image\"),\n",
        "        gr.Radio([\"Foreground\", \"Background\"], label=\"Select Region to Colorize\")\n",
        "    ],\n",
        "    outputs=gr.Image(label=\"Colorized Output\"),\n",
        "    title=\"Task 5: Semantic Segmentation Based Targeted Colorization\",\n",
        "    description=\"Colorizes only selected regions using semantic segmentation.\"\n",
        ")\n",
        "\n",
        "interface.launch(share=True)"
      ]
    }
  ]
}