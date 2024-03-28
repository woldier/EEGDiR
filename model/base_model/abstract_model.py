# -*- coding: utf-8 -*-
"""
@Time ： 3/23/24 12:57 PM
@Auth ： woldier wong
@File ：abstract_model.py
@IDE ：PyCharm
@DESCRIPTION：Base模块, 提供一个抽象model
"""
import torch
from abc import ABC, abstractmethod
from typing import Optional, final


class AbstractDenoiser(torch.nn.Module, ABC):
    """
    一个抽象类, 所有实现的方法都会继承自次class
    ABC是Python标准库中abc模块中的一个类，
    它是Abstract Base Class（抽象基类）的缩写。abstractmethod是abc模块中的一个装饰器函数。
    它们的作用如下：
        - ABC（抽象基类）：抽象基类是一种特殊的类，不能被实例化。它定义了一组方法的接口，
        但并没有实现这些方法的具体功能。抽象基类的主要作用是作为接口规范，
        用于约束子类必须实现某些方法。
        Python中的抽象基类通常用于定义接口规范，而具体的子类则负责实现这些接口规范。
        - abstractmethod（抽象方法装饰器）：抽象方法装饰器用于标记一个方法为抽象方法。
        抽象方法是一种在抽象基类中定义的方法，它只有方法的接口声明而没有具体的实现。
        子类必须实现抽象基类中所有标记为抽象方法的方法，否则在实例化子类对象时会抛出异常。
    """

    def __init__(self, *args, **kwargs):
        super(AbstractDenoiser, self).__init__()
        loss_function = kwargs.pop("loss_function")
        assert loss_function is not None, '''the param "loss_function" must be not None!'''
        self.loss_function = loss_function

    @final
    def forward(
            self, x: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> (torch.Tensor, Optional[torch.Tensor]):
        """
        forward方法, 本方法需要传入输入x以及对应的target
        本方法默认调用self._inner_forward(x), 得到网络的预测结果, 因此user需要自己实现self._inner_forward(x)方法
        在得到预测值pre之后, 会根据当前是否处于training模式或者是否传入了target来决定是否计算loss.
        :param x:
        :param target:
        :return: 本网络 默认返回 预测值pre 和 loss(loss根据策略可能有值可能为空)
        """
        pre = self._inner_forward(x)
        loss = None
        if self.training or target is not None:
            assert target is not None, "model in train mod, the target can't be None!"
            loss = self.compute_loss(pre, target)
        return pre, loss

    @abstractmethod
    def _inner_forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def compute_loss(self, pre: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算损失的方法
        :param pre:
        :param target:
        :return:
        """
        loss = self.loss_function(pre, target)
        return loss
