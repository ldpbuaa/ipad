
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')
import csv
from PIL import Image

class TBSummary(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)
        self.avg_state = {}
        self.csv_buffer = {}  # for csv log

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()
        self.csv_buffer[tag] = value
        self.csv_buffer['global_step'] = step

    def write_to_csv(self, filename):
        file_path = os.path.join(self.log_dir, filename)
        if os.path.exists(file_path):
            with open(file_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(list(self.csv_buffer.values()))
        else:
            with open(file_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(list(self.csv_buffer.keys()))
                writer.writerow(list(self.csv_buffer.values()))

        self.csv_buffer = {}

    def image_summary(self, tag, images, step, is_pil=False):
        """Log a list of images. For colored images only."""
        if is_pil:
            with self.writer.as_default():
                tf.summary.image(tag, images, step=step)
            return
        if len(images.shape) == 4:
            for i, img in enumerate(images):
                img = img.permute((1,2,0))
                img = img[None, :, :, :]
                with self.writer.as_default():
                    tf.summary.image(f'{tag}/{i}', img, step=step)
        else:
            images = images.permute((1,2,0))
            images = images[None, :, :, :]
            with self.writer.as_default():
                tf.summary.image(tag, images, step=step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        values = values.cpu().numpy().flatten()
        with self.writer.as_default():
            tf.summary.histogram(tag, values, step=step, buckets=bins)
