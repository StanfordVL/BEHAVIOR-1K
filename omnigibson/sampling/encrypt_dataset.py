from omnigibson.utils.asset_utils import get_all_object_models, encrypt_file
import os

for model_path in get_all_object_models():
    model_name = os.path.basename(model_path)
    encrypted_usd_path = os.path.join(model_path, "usd", model_name + ".encrypted.usd")
    unencrypted_usd_path = os.path.join(model_path, "usd", model_name + ".usd")

    if not os.path.isfile(encrypted_usd_path) and not os.path.isfile(unencrypted_usd_path):
        print("No USD file found for {}".format(model_name))
        continue

    if not os.path.isfile(encrypted_usd_path) and os.path.isfile(unencrypted_usd_path):
        print("Encrypting {}".format(model_name))
        encrypt_file(unencrypted_usd_path, encrypted_usd_path)

    if os.path.isfile(unencrypted_usd_path):
        print("Removing {}".format(unencrypted_usd_path))
        os.remove(unencrypted_usd_path)
