def test_criterion_minimum():
    phi = [3.0, 2.0, 1.0, 2.0, 3.0]
    d1 = [phi[i+1] - phi[i] for i in range(len(phi)-1)]
    d2 = [d1[i+1] - d1[i] for i in range(len(d1)-1)]
    min_idx = 2
    assert abs(d1[min_idx-1]) < 1e-6 and abs(d1[min_idx]) < 1e-6
    assert d2[min_idx-1] > 0
