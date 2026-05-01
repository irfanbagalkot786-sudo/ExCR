import os
import sys

def test():
    print("Starting import test...")
    try:
        import django
        print(f"Django version: {django.get_version()}")
    except ImportError:
        print("Django not found")
        return

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'MyProject.settings')
    try:
        from django.core.management import execute_from_command_line
        import myapp.views
        print("myapp.views imported successfully")
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
        print(f"Haarcascades path: {cv2.data.haarcascades}")
    except Exception as e:
        print(f"Error during import: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
