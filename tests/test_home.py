import typedload

from deepkit.model import HomeConfig


def test_home_config_convert():
    config = typedload.load({
        'accounts': [
            {'id': '1', 'name': 'peter', 'port': 8080, 'ssl': False, 'username': '', 'host': 'deepkit.ai', 'token': 'abc'},
            {'id': '2', 'name': 'localhost', 'port': 8080, 'ssl': False, 'username': '', 'host': 'deepkit.ai', 'token': 'abc'}
        ],
        'folderLinks': []
    }, HomeConfig)

    assert config.get_account_for_id('1').name == 'peter'
    assert config.get_account_for_id('2').name == 'localhost'
