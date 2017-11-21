创建数据命令：python tools/generate_font_image.py
将测试数据转换为tf文件：python tools/generate_tfrecord.py --txt_input=dataset/test.txt  --output_path=dataset/test.record
将训练数据转换为tf文件：python tools/generate_tfrecord.py --txt_input=dataset/train.txt  --output_path=dataset/train.record
