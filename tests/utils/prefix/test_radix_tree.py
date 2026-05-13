from lumilake.utils.prefix.radix_tree import Placeholder, TemplatedRadixTree


def test_templated_radix_tree_accepts_shorter_placeholder_key() -> None:
    tree = TemplatedRadixTree()
    long_key = ("abc", Placeholder("dep"), "tail")
    short_key = ("abc", Placeholder("dep"))

    tree.add(long_key, "worker", "n1")
    tree.add(short_key, "worker", "n2")

    assert set(tree.node_map) == {"n1", "n2"}
    assert tree.node_map["n1"].get_prefix() == long_key
    assert tree.node_map["n2"].get_prefix() == short_key


def test_templated_radix_tree_remove_shorter_placeholder_key() -> None:
    tree = TemplatedRadixTree()
    long_key = ("abc", Placeholder("dep"), "tail")
    short_key = ("abc", Placeholder("dep"))

    tree.add(long_key, "worker", "n1")
    tree.add(short_key, "worker", "n2")
    tree.remove("n2")

    assert set(tree.node_map) == {"n1"}
    assert tree.node_map["n1"].get_prefix() == long_key
