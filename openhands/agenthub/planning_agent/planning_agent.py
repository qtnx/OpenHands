from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.message import Message, TextContent
from openhands.events.action import Action, AgentFinishAction, MessageAction
from openhands.events.observation import UserRejectObservation
from openhands.llm.llm import LLM
from openhands.core.logger import openhands_logger as logger

class PlanningAgent(Agent):
    VERSION = '1.0'

    def __init__(self, llm: LLM, config: AgentConfig):
        super().__init__(llm, config)
        self.plan_approved = False
        self.plan = None
        self.waiting_for_feedback = False

    def step(self, state: State) -> Action:
        # Kiểm tra nếu đang đợi phản hồi
        if self.waiting_for_feedback:
            last_user_message = state.get_last_user_message()
            if last_user_message:
                self.waiting_for_feedback = False
                return self._handle_user_feedback(last_user_message, state)
            return MessageAction(
                content="Đang chờ phản hồi của bạn về kế hoạch...",
                wait_for_response=True
            )

        if not self.plan_approved:
            if not self.plan:
                # Tạo kế hoạch chi tiết
                plan_prompt = (
                    "Trước khi tiến hành, hãy tạo một kế hoạch thực thi chi tiết bao gồm:\n"
                    "1. Cách tiếp cận tổng thể để giải quyết nhiệm vụ\n"
                    "2. Các bước thực hiện cụ thể\n"
                    "3. Những rủi ro hoặc cân nhắc tiềm ẩn\n"
                    "4. Kết quả mong đợi ở mỗi bước\n"
                    "Vui lòng trình bày kế hoạch này để xem xét trước khi tiến hành thực thi."
                )
                messages = [Message(role='user', content=[TextContent(text=plan_prompt)])]
                response = self.llm.completion(messages=self.llm.format_messages_for_llm(messages))
                self.plan = response.choices[0].message.content

                # Yêu cầu người dùng phê duyệt kế hoạch
                self.waiting_for_feedback = True
                return MessageAction(
                    content=(
                        f"Kế hoạch thực thi đề xuất:\n\n{self.plan}\n\n"
                        "Vui lòng xem xét kế hoạch này và phản hồi với:\n"
                        "- 'approve' để tiến hành thực thi\n"
                        "- 'modify <gợi ý>' để yêu cầu chỉnh sửa kế hoạch\n"
                        "- 'reject' để hủy bỏ thực thi"
                    ),
                    wait_for_response=True
                )

        # Tiếp tục thực thi sau khi kế hoạch được phê duyệt
        return self._execute_plan(state)

    def _handle_user_feedback(self, feedback: str, state: State) -> Action:
        """Xử lý phản hồi của người dùng về kế hoạch"""
        feedback = feedback.lower().strip()
        
        if "approve" in feedback:
            self.plan_approved = True
            return MessageAction(
                content="Kế hoạch đã được phê duyệt. Tiến hành thực thi.",
                wait_for_response=False
            )
        elif "reject" in feedback:
            self.plan = None
            return AgentFinishAction(
                thought="Kế hoạch đã bị từ chối. Kết thúc thực thi."
            )
        else:
            # Xử lý yêu cầu chỉnh sửa
            modification_request = feedback.replace("modify", "").strip()
            self.plan = None
            
            # Tạo kế hoạch mới dựa trên phản hồi
            plan_prompt = (
                f"Dựa trên phản hồi sau của người dùng:\n{modification_request}\n\n"
                "Hãy tạo một kế hoạch thực thi mới chi tiết bao gồm:\n"
                "1. Cách tiếp cận tổng thể để giải quyết nhiệm vụ\n"
                "2. Các bước thực hiện cụ thể\n"
                "3. Những rủi ro hoặc cân nhắc tiềm ẩn\n"
                "4. Kết quả mong đợi ở mỗi bước"
            )
            messages = [Message(role='user', content=[TextContent(text=plan_prompt)])]
            response = self.llm.completion(messages=self.llm.format_messages_for_llm(messages))
            self.plan = response.choices[0].message.content

            # Yêu cầu phê duyệt cho kế hoạch mới
            self.waiting_for_feedback = True
            return MessageAction(
                content=(
                    f"Kế hoạch thực thi đã được chỉnh sửa:\n\n{self.plan}\n\n"
                    "Vui lòng xem xét kế hoạch này và phản hồi với:\n"
                    "- 'approve' để tiến hành thực thi\n"
                    "- 'modify <gợi ý>' để yêu cầu chỉnh sửa thêm\n"
                    "- 'reject' để hủy bỏ thực thi"
                ),
                wait_for_response=True
            )

    def _execute_plan(self, state: State) -> Action:
        """Thực thi kế hoạch sau khi được phê duyệt"""
        # TODO: Implement actual execution logic here
        # For now, just return a finish action
        return AgentFinishAction(
            thought="Kế hoạch đã được thực thi hoàn tất."
        )

    def reset(self) -> None:
        """Reset agent state"""
        super().reset()
        self.plan_approved = False
        self.plan = None
        self.waiting_for_feedback = False