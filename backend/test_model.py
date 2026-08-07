"""Smoke-test the persisted landmark model without synthetic samples."""

from services.translator_model import TranslatorModel


def test_model_loads():
    translator = TranslatorModel()
    assert translator.mode == 'deep_learning', 'No trained TensorFlow model is available.'
    assert translator.model.input_shape[-1] == 126, 'Model input must remain 126 landmarks.'
    print(f'Model loaded: {translator.mode}; validation_accuracy={translator.metadata.get("val_accuracy")}')


if __name__ == '__main__':
    test_model_loads()
