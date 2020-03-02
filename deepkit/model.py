import os
from typing import NamedTuple, Optional, List


class ExperimentOptions(NamedTuple):
    """
    Per default the account linked to this folder is used (see `deepkit link`), this is on a new system `localhost`. 
    You can overwrite which account is used by specifying the name here (see `deepkit id` for 
    available accounts in your system). 
    """
    account: Optional[str] = None

    """
    Per default the project linked to this folder is used (see `deepkit link`). 
    You can overwrite which proect is used. 
    Names is format of either `my-project`, or `user/my-project`, or `org/my-project`.
    
    If the current folder is not linked and you don't specify a project here, an error is raised since
    Deepkit isn't able to know to which project the experiments data should be sent.
    """
    project: Optional[str] = None


class Account(NamedTuple):
    id: str
    port: int
    ssl: bool
    username: str
    token: str
    host: str
    name: str


class FolderLink(NamedTuple):
    accountId: str
    name: str
    path: str
    projectId: str


class HomeConfig(NamedTuple):
    accounts: List[Account]
    folderLinks: List[FolderLink]

    def get_account_for_name(self, name: str) -> Account:
        for account in self.accounts:
            if account.name == name:
                return account
        raise Exception(f'No account for name {name} configured. Use `deepkit login` to add new accounts.')

    def get_account_for_id(self, id: str) -> Account:
        for account in self.accounts:
            if account.id == id:
                return account
        raise Exception(f'No account for id {id}')

    def get_folder_link_of_directory(self, dir: str) -> FolderLink:
        link_map = {}
        for item in self.folderLinks:
            link_map[item.path] = item

        while dir not in link_map:
            dir = os.path.realpath(os.path.join(dir, '..'))
            if dir == os.path.realpath(os.path.join(dir, '..')):
                # reached root
                break

        if dir in link_map:
            return link_map[dir]

        raise Exception('No project linked for folder ' + dir)
