# Блок-схема текущего алгоритма
![Блок-схема](https://gitverse.ru/eugenykondratyev/python_for_bmstu_students/content/main/dz/phase_portrait/docs/Pasted%20image%2020241010225013.png)
На данный момент в программе реализован следующий функционал:
1) Считывание исходных данных из `xls` файла (`parser_data.py`);
2) Генерация объекта (`initial_data_class.py`) управления (`ControlObject`) и объекта системы управления(`MotionControlSystem`). На этом этапе также рассчитываются возмущающие моменты (`ComputeMoments`), действующие на аппарат;
3) Решение уравнений (`AnalyticSolver`) плоского вращательного движения обособленно в каждом канале методом фазовой плоскости (`PhasePlane`).