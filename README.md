# Tuvalon_group_9

## 任务分工

- MerlinStrategy + PercivalStrategy:
- AssassinStrategy + MorganaStrategy:
- KnightStrategy + 代码统筹(辅助函数+RoleStrategy基类+Player类) + 测试:
- OberonStrategy + report编写:

## 代码结构说明

服务器直接调用Player类。Player类中和角色相关的方法，如walk(),say(),mission_vote1(),mission_vote2(),assass()等，统一放在相应的角色策略类中，6个角色策略类均继承自RoleStrategy基类。为使代码便于阅读，所有不被服务器直接调用的辅助函数全部在Player类之外定义。

## 策略说明

### MerlinStrategy

### PercivalStrategy

### AssassinStrategy

### MorganaStrategy

### KnightStrategy

### OberonStrategy
