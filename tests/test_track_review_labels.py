"""The tracked list keeps original assignments distinct from reviewer decisions."""
from multiphy import clusters
from tests.test_viewer_tables import _day_table_panel, qt


def test_tracked_review_labels_keep_backend_identity(qt):
    panel, controller, _ = _day_table_panel()
    original_id = clusters.cluster_id(['A-1', 'B-1'])

    def row_for(track_id):
        from multiphy.viewer.qt import QtCore
        return next(i for i in range(panel.table.rowCount())
                    if panel.table.item(i, 0).data(QtCore.Qt.UserRole) == track_id)

    try:
        i = row_for(original_id)
        assert panel.table.item(i, 4).text() == 'unreviewed'
        original_label = panel.table.item(i, 5).text()
        assert original_id in panel.table.item(i, 5).toolTip()
        controller.do('accept', cluster_id=original_id)
        panel.refresh_list()
        i = row_for(original_id)
        assert panel.table.item(i, 4).text() == 'accepted'
        assert panel.table.item(i, 5).text() == original_label

        controller.do('remove_member', cluster_id=original_id, members=['B-1'])
        panel.refresh_list()
        for member in ['A-1', 'B-1']:
            i = row_for(clusters.cluster_id([member]))
            assert panel.table.item(i, 4).text() == 'unreviewed'
            assert panel.table.item(i, 5).text() == original_label
            assert original_id in panel.table.item(i, 5).toolTip()
    finally:
        panel.close()
