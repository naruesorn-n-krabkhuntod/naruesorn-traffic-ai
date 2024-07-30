# iplus-traffic-ai

### setup project
1. install requirement python lib
<code>pip install -r requirements.txt </code>
2. download sample video and paste in <code> ./assets/vodeo/ </code>
[ sample.mp4 ](https://drive.google.com/file/d/17578YcnAcKQsaAGuuPao1PCuIBuedNyf/view?usp=sharing)
## Note
if found this pytorch error (Commonly found on devices that do not have NVDIA GPUs.)

<code> OSError: [WinError 126] The specified module could not be found. Error loading .../fbgemm.dll or one of its dependencies </code>

install anaconda or miniconda and run this command in anaconda prompt
<code> conda install pytorch torchvision cpuonly -c pytorch </code>


