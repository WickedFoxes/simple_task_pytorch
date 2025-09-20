from tqdm import tqdm
import random, time
from src.logger.base import LoggerBase
from src.registry import register

@register("logger", "tqdm")
class TqdmLogger(LoggerBase):
    """
    start_epoch 없이 __init__ 설정만으로 사용하는 진행바 로거.
    - epoch 자동 감지: metrics dict에 epoch 키가 있으면 값 변화 시 새 pbar 생성
    - total 결정: __init__의 total 인자, 또는 metrics['total_steps']로 최초 1회 설정
    """
    def __init__(
        self,
        desc: str = "train",
        total: int | None = None,
        leave: bool = False,
        epoch_key: str = "epoch",
        total_steps_key: str = "total_steps",
        disable: bool = False,
        mininterval: float = 0.1,   # tqdm 갱신 최소 간격
    ):
        from tqdm import tqdm
        self._tqdm = tqdm
        self.desc = desc
        self.leave = leave
        self.epoch_key = epoch_key
        self.total_steps_key = total_steps_key
        self.disable = disable
        self.mininterval = mininterval

        self.params = {}
        self.current_epoch = None
        self.total = total           # 명시 안 하면 None → 무한 진행 스타일
        self.pbar = None

    # --- 내부 유틸 ---
    def _open_pbar(self, epoch_label: str | None):
        if self.pbar:  # 안전 닫기
            self.pbar.close()
        desc = self.desc if epoch_label is None else f"{self.desc} | epoch {epoch_label}"
        self.pbar = self._tqdm(
            total=self.total,
            desc=desc,
            leave=self.leave,
            disable=self.disable,
            dynamic_ncols=True,
            mininterval=self.mininterval,
        )

    # --- 공용 API ---
    def log_metrics(self, d: dict, step: int | None = None):
        """
        d: 기록할 메트릭 딕셔너리 (예: {"loss": 0.123, "acc": 0.9, "epoch": 3, "total_steps": 1000})
        step: 외부에서 관리하는 글로벌/배치 step (진행바 갱신에는 사용 안 함)
        """
        # total이 미정이고, 메트릭에 total_steps가 처음 들어오면 total 설정
        if self.total is None and self.total_steps_key in d and isinstance(d[self.total_steps_key], int):
            self.total = int(d[self.total_steps_key])
            # 이미 pbar가 열려있다면 total 업데이트
            if self.pbar is not None:
                self.pbar.total = self.total
                self.pbar.refresh()

        # epoch 자동 감지
        epoch_in = d.get(self.epoch_key, None)
        if self.pbar is None:
            # 처음 호출 시 진행바 오픈
            self.current_epoch = epoch_in
            self._open_pbar(epoch_label=str(epoch_in) if epoch_in is not None else None)
        elif epoch_in is not None and epoch_in != self.current_epoch:
            # epoch 변경 감지 시 새 진행바
            self.current_epoch = epoch_in
            self._open_pbar(epoch_label=str(epoch_in))

        # 진행바 후면의 postfix 갱신
        # 너무 많은 키는 시인성 떨어질 수 있으니 주요 값 위주가 좋습니다.
        if self.pbar:
            # 숫자/짧은 문자열만 간단히 표시
            postfix = {k: (f"{v:.4f}" if isinstance(v, (int, float)) else v)
                       for k, v in d.items() if k not in (self.total_steps_key,)}
            self.pbar.set_postfix(postfix, refresh=False)
            self.pbar.update(1)

    def log_params(self, d: dict):
        """하이퍼파라미터/설정 기록 (한 번에 누적해서 보관, 콘솔에도 1회 출력)"""
        self.params.update(d)
        # 과도한 중복 출력 방지: 최근 기록분만 요약 출력
        preview = {k: self.params[k] for k in list(self.params)[-10:]}  # 마지막 10개만 미리보기
        print("[TqdmLogger] params updated (preview):", preview)

    def finalize(self):
        if self.pbar:
            self.pbar.close()
        # 전체 파라미터 최종 요약
        if self.params:
            print("[TqdmLogger] final params:", self.params)
        print("[TqdmLogger] finished.")


