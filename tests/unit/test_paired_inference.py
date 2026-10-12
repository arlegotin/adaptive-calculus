"""Fresh target seeds, never chunks or timing repeats, determine paired inference."""
import math

import pytest
from scipy.stats import t


def analyze(manifest, values, statuses=None):
    from ralup.analysis.confirm import PairedResults, analyze_confirmation
    rows=[]
    for index,seed in enumerate(manifest.payload['seeds']):
        for arm in ('C','Bstar','LR'):
            status=(statuses or {}).get((seed,arm),'completed')
            rows.append({'seed':seed,'arm':arm,'unit':f'fresh-{seed}',
                         'status':status,'bpb':values[arm][index] if status=='completed' else None})
    return analyze_confirmation(PairedResults(manifest.sha256,rows),manifest)


def test_exact_simultaneous_t_and_ratio(e04_factory):
    _,manifest=e04_factory(n=4)
    values={'C':[7,7.1,7.2,7.3],'Bstar':[8,8,8,8],'LR':[8.2,8.2,8.2,8.2]}
    result=analyze(manifest,values)
    delta=[c-b for c,b in zip(values['C'],values['Bstar'])]
    mean=sum(delta)/4
    sd=math.sqrt(sum((x-mean)**2 for x in delta)/3)
    radius=t.ppf(.9875,3)*sd/2
    assert result['n']==4
    assert result['contrasts']['Bstar']['interval']==pytest.approx([mean-radius,mean+radius])
    assert result['contrasts']['LR']['t_critical']==pytest.approx(t.ppf(.9875,3))
    assert result['ratio_vs_B']==pytest.approx(2**mean)
    assert result['contrasts']['Bstar']['ratio_interval']==pytest.approx([2**(mean-radius),2**(mean+radius)])
    assert result['H1_H2_supported'] is True
    assert result['scientific_confirmation_established'] is False


@pytest.mark.parametrize('kind,expected',[('success','supported'),('null','negative'),('inconclusive','inconclusive'),('one','inconclusive')])
def test_two_contrasts_decide_scoped_outcome(e04_factory,kind,expected):
    _,manifest=e04_factory(n=4)
    candidate={'success':[7]*4,'null':[9]*4,'inconclusive':[6,9,6,9],'one':[7]*4}[kind]
    lr=[7]*4 if kind=='one' else [8]*4
    result=analyze(manifest,{'C':candidate,'Bstar':[8]*4,'LR':lr})
    assert result['status']==expected
    assert result['H1_H2_supported'] is (kind=='success')
    assert result['contrasts']['Bstar']['interval'] is not None


def test_constant_differences_are_math_fixture_not_pilot_authority(e04_factory):
    _,manifest=e04_factory(n=4)
    result=analyze(manifest,{'C':[7]*4,'Bstar':[8]*4,'LR':[8]*4})
    assert result['contrasts']['Bstar']['interval']==[-1,-1]
    assert result['contrasts']['Bstar']['degenerate_variance'] is True
    assert result['scientific_confirmation_established'] is False


def test_divergence_retained_with_conditional_quality_and_ordinal(e04_factory):
    _,manifest=e04_factory(n=4)
    result=analyze(manifest,{'C':[7]*4,'Bstar':[8]*4,'LR':[8]*4},{(100,'C'):'diverged'})
    assert result['n']==4
    assert result['all_run_bpb_claim_established'] is False
    assert result['contrasts']['Bstar']['interval'] is None
    assert result['success_conditioned_quality']['Bstar']['n']==3
    assert result['failures']['C']['divergence_probability']==.25
    assert result['ordinal_sensitivity']['Bstar']['losses']==1


def test_missing_nonfinite_duplicates_and_pooling_do_not_establish_claim(e04_factory):
    from ralup.analysis.confirm import PairedResults,analyze_confirmation
    _,manifest=e04_factory(n=4)
    rows=[{'seed':seed,'arm':arm,'unit':f'fresh-{seed}','status':'completed','bpb':7.0 if arm=='C' else 8.0}
          for seed in manifest.payload['seeds'] for arm in manifest.payload['arms']]
    missing=analyze_confirmation(PairedResults(manifest.sha256,rows[:-1]),manifest)
    assert missing['n']==4 and not missing['all_run_bpb_claim_established']
    rows[0]['bpb']=float('nan')
    assert not analyze_confirmation(PairedResults(manifest.sha256,rows),manifest)['all_run_bpb_claim_established']
    rows[0]['bpb']=7.0
    with pytest.raises(ValueError,match='Duplicate'):
        analyze_confirmation(PairedResults(manifest.sha256,rows+rows[:1]),manifest)
    rows[0]['seed']=999
    with pytest.raises(ValueError,match='Unplanned'):
        analyze_confirmation(PairedResults(manifest.sha256,rows),manifest)


def test_dropping_failed_seed_and_reused_units_rejected(e04_factory):
    from ralup.analysis.confirm import PairedResults,analyze_confirmation
    _,manifest=e04_factory(n=4)
    rows=[{'seed':seed,'arm':arm,'unit':f'fresh-{seed}','status':'completed','bpb':7.0}
          for seed in manifest.payload['seeds'][1:] for arm in manifest.payload['arms']]
    result=analyze_confirmation(PairedResults(manifest.sha256,rows),manifest)
    assert result['n']==4 and result['status']=='incomplete'
    for row in rows: row['unit']='reused-warm-parent'
    with pytest.raises(ValueError,match='independent'):
        analyze_confirmation(PairedResults(manifest.sha256,rows),manifest)


def test_predeclared_seed_bootstrap_sign_tied_rank_sensitivities(e04_factory):
    _,manifest=e04_factory(n=4)
    result=analyze(manifest,{'C':[7]*4,'Bstar':[8]*4,'LR':[8]*4})
    sensitivity=result['sensitivities']['Bstar']
    assert sensitivity['sign']['p_two_sided']==.125
    assert sensitivity['rank']['p_two_sided']==.125
    assert sensitivity['bootstrap']['interval']==[-1,-1]
    assert result['sensitivities']['role']=='predeclared sensitivity only'


def test_timing_repetitions_cannot_change_statistical_n(e04_factory):
    from ralup.analysis.confirm import PairedResults,analyze_confirmation
    _,manifest=e04_factory(n=4)
    rows=[{'seed':seed,'arm':arm,'unit':f'fresh-{seed}','status':'completed','bpb':7.0 if arm=='C' else 8.0,
           'timing_observations':[.2]*1000} for seed in manifest.payload['seeds'] for arm in manifest.payload['arms']]
    report=analyze_confirmation(PairedResults(manifest.sha256,rows),manifest)
    assert report['n']==4 and report['contrasts']['Bstar']['n']==4
    assert report['sensitivities']['joint_complete_n']==4
