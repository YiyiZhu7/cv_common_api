# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

if __name__ == '__main__':
    from deploy_plate.flask_app import run

    run()
