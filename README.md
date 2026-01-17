# MAX7800x jupyter training
![ZHAW INES](/assets/zhaw-ines-rgb.png)
# About
For the past several months, I've been deep in the trenches with the [ai8x-training](https://github.com/analogdevicesinc/ai8x-training) tool, honing my skills in training various Convolutional Neural Network (CNN) architectures specifically tailored for the MAX78000 and MAX78002 devices. Yet, from the outset, the [ai8x-training](https://github.com/analogdevicesinc/ai8x-training) tool often felt more like a roadblock than an enabler.

Among the myriad challenges I encountered, the inability to make real-time adjustments, fine-tune models while freezing or unfreezing specific layers, and transferring custom weight sets proved to be major pain points. These limitations stifled the efficiency and flexibility I needed.

I aim to empower you with the knowledge of how to train the MAX78000 and MAX78002 devices right from your Jupyter notebook. With this approach, you'll break free from the shackles of real-time debugging and fine-tuning woes, ensuring seamless interaction with your neural networks. Let's dive into the nitty-gritty of this process and unlock the full potential of these devices.  

In this guide, we'll walk you through **training MAX7800x directly from a Jupyter notebook**. The best part? You won't need the [ai8x-training](https://github.com/analogdevicesinc/ai8x-training) tool for this process.

Please ensure you are using a Linux environment, as this guide is tailored for Linux users. Additionally, this method has been tested exclusively with Nvidia GPUs. Make sure you have the following prerequisites in place:

- Python 3.8
- PyTorch 2.0.1

It's important to note that this tutorial doesn't substitute the conventional method of synthesizing the network. You'll still need the official tool [ai8x-synthesis](https://github.com/analogdevicesinc/ai8x-training) for this critical step.

The key distinction here is that you now have the flexibility to train your network directly from your Jupyter notebook, bypassing the need to rely on the MAX78 training tool. To see a practical example of the entire process, from training to synthesis, check out the **"How to get started"** chapter.

## What is included
- MNIST classifier for the MAX78
- QAT (Quantization Aware Training)
- Export KAT (known-answer test)
- Export QAT model (.pth.tar), compatible with the [synthesis tool](https://github.com/analogdevicesinc/ai8x-training)

## Directory structure:
```
.
├── README.md
├── assets
├── data            <-- MNIST dataset storage
├── distiller       <-- distiller submodule, required for QAT
├── max78_modules   <-- max78 python modules
│   ├── ai8x.py
│   └── devices.py
├── requirements-cu11.txt   <-- python requirements
└── train_MNIST.ipynb   <-- example Jupyter notebook
```

## How to get started
Let's dive into the world of training the MAX7800x, right from the comfort of your Jupyter notebook.

### Setup
- Install anaconda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
- First also activate it 
```
eval "$(/root/anaconda3/bin/conda shell.bash hook)"
```
- Create an environment and activate it:
```
conda create -n max78-training-jupyter python=3.8
conda activate max78-training-jupyter
```

- Clone the repo and checkout the submodules
```
git clone --recurse-submodules https://github.com/InES-HPMM/MAX7800x-Jupyter-training
```

- Install the Python requirements
Had so much problem with the pycocotools so did it like this
```
 conda install -c conda-forge pycocotools
```
- In requirements i removed the pycocotools
```
pip install -r requirements-cu11.txt
```

- Everything should be ready to go! Open the notebook: `train_MNIST.ipynb`

### Training the Network
To train the network, follow these steps within the `train_MNIST.ipynb` notebook:
- Execute the notebook cells in sequential order.
- The notebook will download the MNIST dataset, train the network, and export the KAT and the QAT model.
- The QAT model should have an accuracy of ~99% on the test set

Take a look at the image below to get a visual idea of what to expect during QAT training:
![QAT max78 notebook](/assets/qat_training.png)

Finally, confirm that the following files have been generated in the `max78_jupyter_training` directory:
- `sample_mnist_2828.npy`
- `qat_class_mnist_checkpoint.pth.tar`

### Synthesizing the Network
- **IMPORTANT**: It's crucial to ensure that you have a functional installation of the [ai8x-synthesis](https://github.com/analogdevicesinc/ai8x-training) tool. If you haven't done so yet, please refer to the official GitHub repository and follow the provided instructions for installation.

- **Note**: The following steps are not unique, but are standard for synthesizing any trained network using the ai8x-synthesis tool.

## More instructions
eval "$(/mnt/HC_Volume_102266423/anaconda3/bin/conda shell.bash hook)"
conda create -n ai8x-synthesis python=3.11.8 -y
conda activate ai8x-synthesis
cd /mnt/HC_Volume_102266423/ai8x-synthesis
TMPDIR=/mnt/HC_Volume_102266423/tmp pip install --no-cache-dir -r requirements.txt
python3 quantize.py custom-mnist/qat_class_mnist_checkpoint.pth.tar custom-mnist/qat_class_mnist_checkpoint_q8.pth.tar --device MAX78002 -v


#### Prepare the KAT
- Copy the `sample_mnist_2828.npy` file to the `ai8x-synthesis/tests` directory
#### Prepare the QAT model
- Make a subfolder in the `ai8x-synthesis/` directory, for example `ai8x-synthesis/custom-mnist`
- Copy the `qat_class_mnist_checkpoint.pth.tar` file to the `custom-mnist` directory
- Quantize the network: `python quantize.py custom-mnist/qat_class_mnist_checkpoint.pth.tar custom-mnist/qat_class_mnist_checkpoint_q8.pth.tar --device MAX78002 -v`
#### Populate the network descriptor file
- Create `custom-mnist/classifier.yaml` and insert:
(make sure to not forget the **empty line at the end of the file**)
```
---
# HWC (little data) configuration for MNIST
# MNIST classifier Model
# Compatible with MAX78002

arch: ai87net-mnist-classifier
dataset: mnist_2828

layers:
  # Layer 0: step1
  - in_offset: 0xA000
    out_offset: 0x0000
    processors: 0x0000000000000001
    output_processors: 0x00000000ffffffff
    operation: conv2d
    kernel_size: 3x3
    pad: 1
    activate: ReLU
  # Layer 1: step2
  - out_offset: 0xA000
    processors: 0x00000000ffffffff
    output_processors: 0xffffffffffffffff
    operation: conv2d
    kernel_size: 3x3
    pad: 1
    max_pool: 2
    pool_stride: 2
    activate: ReLU
  # Layer 2: step3
  - out_offset: 0x0000
    processors: 0xffffffffffffffff
    output_processors: 0x0000ffffffffffff
    operation: conv2d
    kernel_size: 3x3
    pad: 1
    max_pool: 2
    pool_stride: 2
    activate: ReLU
  # Layer 3: pool3
  - out_offset: 0xA000
    processors: 0x0000ffffffffffff
    output_processors: 0x0000ffffffffffff
    operation: Passthrough
    max_pool: 2
    pool_stride: 2
    activate: None
  # Layer 4: fc1
  - out_offset: 0x0000
    processors: 0x0000ffffffffffff
    output_processors: 0xffffffffffffffff
    operation: mlp
    flatten: true
    activate: ReLU
  # Layer 5: fc2
  - out_offset: 0xA000
    processors: 0xffffffffffffffff
    output_processors: 0x00000000000003ff
    operation: mlp
    activate: None
    output_width: 32

```

#### Synthesize the network, as you would normally do:
```
python ai8xize.py --test-dir CNN_example --prefix ai87net-mnist-classifier --checkpoint-file custom-mnist/qat_class_mnist_checkpoint_q8.pth.tar --config-file custom-mnist/classifier.yaml --device MAX78002 --timer 0 --display-checkpoint --verbose --softmax
```

Lastly, within the `CNN_example/ai87net-mnist-classifier` directory, you should find the generated CNN files.


#### This is additional that didnt was here from the start
```
python ai8xize.py \
  --test-dir CNN_example \
  --prefix ai87net-mnist-classifier \
  --checkpoint-file custom-mnist/qat_class_mnist_checkpoint_q8.pth.tar \
  --config-file custom-mnist/classifier.yaml \
  --device MAX78002 \
  --sample-input tests/sample_mnist_2828.npy \
  --verify-kernels \
  --timer 0 --display-checkpoint --verbose --softmax --overwrite
  ```

#### Now when this is ready we should deploy the CNN-model to the MAX78002EVKIT
- The model is now in the CNN_example folder
- Start Visual Studio Code and choose CodeFusion Studio 
- Choose New Workspace
- Search for MAX78002, continue
- Choose EVKIT-V1, continue
- MSDK Single Core PyTorch AI Model using CNN, continue
- Choose ARM only, continue
- Choose MSDK, ai8xize is made for this, continue
- Keep the flags, continue
- Now the workspace is ready - its time to copy the files you have done in the previous steps 
- The files that are needed that should be copy is the following into the src folder:
```
YourProject/
├── main.c
├── cnn.c
├── cnn.h
├── weights.h
├── sample_mnist_2828.h
├── softmax.c
└── Include/  (SDK includes - auto-added)
```
- Build it with the build icon in the bottom left corner
- If all went succesful you should have a <project.elf> file created, this should be put on the board
- To program the MAX78002EVKIT
- Use the A MAX32625PICO adapter that followed along the board
- Connect it to “SWD” debug header (JH8)
- Connect it to the computer
- Run the Compiler with debug F5 or in the top.

- There were a lot of problems that didnt make me flash my MAX78002EVKIT: 
Your MAX78002EVKIT is now successfully flashed and running! Here's what was wrong and what I fixed:

The Problem:
Your .vscode/settings.json was missing critical configuration paths needed for flashing:

cfs.openocd.path - Path to OpenOCD debugger
cfs.toolchain.armAArch32GCC.path - Path to ARM GCC toolchain
cfs.tool.path.msdk - Path to MaximSDK
What I Fixed:
✅ Added missing paths to settings.json:

OpenOCD: OpenOCD
ARM GCC: 10.3
MSDK: MaximSDK
✅ Updated flash.gdb to use full OpenOCD path

✅ Flashed your firmware (355,696 bytes) via OpenOCD + CMSIS-DAP

✅ Reset and started your device

Next Steps:

- Its also important to use the System Setup in CFS to put in the UART if you want output in the terminal screen
- After the System setup in CFS you need to generate new code in the last section 
- After that save it
- Clean build and flash it again 
- This is the command to flash the chip: $env:Path = "C:\MaximSDK\Tools\OpenOCD;$env:Path"; C:\MaximSDK\Tools\GNUTools\10.3\bin\arm-none-eabi-gdb.exe --cd="C:\Users\henri\cfs\2.0.1\TestCNN-new\m4" --se="C:\Users\henri\cfs\2.0.1\TestCNN-new\m4\build\m4.elf" --symbols=C:\Users\henri\cfs\2.0.1\TestCNN-new\m4\build\m4.elf -x="C:\Users\henri\cfs\2.0.1\TestCNN-new\m4\.vscode\flash.gdb" --ex="flash_m4_run C:/MaximSDK/Tools/OpenOCD C:/MaximSDK/Tools/OpenOCD/scripts/interface/cmsis-dap.cfg C:/MaximSDK/Tools/OpenOCD/scripts/target/max78002.cfg" --batch