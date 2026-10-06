def mean: add/length;
. as $m
| ($m.rows | group_by(.id)) as $roots
| def cond($c): [ $m.rows[] | select(.condition==$c) ];
  def loss($c): cond($c) | map(.set_loss) | mean;
  def corrected_ids: [ $roots[] | select((.[0].baseline_correct|not) and ([.[]|select(.condition=="real")|.correct]|all)) | .[0].id ];
  def harmed_ids: [ $roots[] | select((.[0].baseline_correct) and ([.[]|select(.condition=="real")|.correct]|all|not)) | .[0].id ];
  def rev($c): corrected_ids as $ids | [ $roots[] | select(.[0].id as $i | $ids|index($i)) | select([.[]|select(.condition==$c)|.correct]|any|not) ] | length;
  { update:$m.update, arm:$m.arm, n_rows:($m.rows|length), roots:($roots|length),
    real_loss:loss("real"), corrected:(corrected_ids|length), harmed:(harmed_ids|length),
    contrast: ([ "shuffle_complete_e002","successor_only_e002","successor_only_e402","successor_only_e502" ] | map({(.): {c:(loss(.)-loss("real")), reversed:rev(.)}}) | add),
    allnull_exact: ($m.rows | map(select(.condition=="all_null") | [.candidates[] | (.baseline_logit==.final_logit)] | all) | all),
    board_helpful: (corrected_ids as $ids | [ $roots[] | select(.[0].id as $i | $ids|index($i)) | select([.[]|select(.condition=="board_flags_zero")|.correct]|all) ] | length),
    succ_helpful: (corrected_ids as $ids | [ $roots[] | select(.[0].id as $i | $ids|index($i)) | select([.[]|select(.condition=="successor_frames_zero")|.correct]|all) ] | length) }
