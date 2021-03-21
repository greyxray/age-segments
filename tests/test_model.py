from model.build import build_model


def test_handles_model(data):
    # Check that the pipeline works
    model = build_model().fit(data, data["age"])
    model.predict(data)
