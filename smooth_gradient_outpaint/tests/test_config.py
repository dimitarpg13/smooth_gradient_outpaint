import unittest
from smooth_gradient_outpaint.config import OutpainterConfig


class OutpainterConfigTestCase(unittest.TestCase):
    def setUp(self):
        self.config = OutpainterConfig().config

    def test_simple_fill_config(self):
        config2 = OutpainterConfig().config
        config2["outpainter"]["blur"]["parameters"]["size"] = 5
        self.assertTrue(self.config["outpainter"]["blur"]["parameters"]["size"] ==
                        config2["outpainter"]["blur"]["parameters"]["size"])


if __name__ == "__main__":
    unittest.main()
