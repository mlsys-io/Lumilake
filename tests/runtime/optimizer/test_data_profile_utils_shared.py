from lumilake.runtime.data_profile_utils import (
    coerce_data_profile_footprints,
    data_profile_key_for_node_query,
    normalize_table_name,
)
from lumilake.runtime.optimizer.halo import HaloOptimizer


def test_normalize_table_name_strips_quotes_and_whitespace() -> None:
    assert normalize_table_name(' "public"."orders" ; ') == "public.orders"


def test_data_profile_key_for_node_query_uses_shared_format() -> None:
    assert (
        data_profile_key_for_node_query("node_a", "node_a_query")
        == "data_profile::node_a::node_a_query"
    )


def test_coerce_data_profile_footprints_filters_invalid_values() -> None:
    value = {
        "tbl_a": "3",
        "tbl_b": 0,
        "tbl_c": -1,
        "tbl_d": "bad",
        5: 9,
    }
    assert coerce_data_profile_footprints(value) == {"tbl_a": 3}


def test_data_profile_utils_and_halo_use_same_table_normalization() -> None:
    optimizer = HaloOptimizer()
    template = 'SELECT * FROM "public"."orders" WHERE id = {id};'
    assert normalize_table_name('"public"."orders"') == "public.orders"
    assert optimizer._extract_sql_table(template) == "public.orders"
