from xirt import basic


def test_add_numbers():
    """
    Test addition of two numbers.
    :return:
    """
    result = basic.add_numbers(10, 30)
    assert result == 40
