+++
date = '2025-05-18T11:30:19+08:00'
draft = false
title = 'ts-foundation'
+++

# TypeScript 基础教程

## 目录
1. [TypeScript 简介](#typescript-简介)
2. [基础类型](#基础类型)
3. [接口与类型](#接口与类型)
4. [函数](#函数)
5. [类](#类)
6. [泛型](#泛型)
7. [高级类型](#高级类型)
8. [模块](#模块)
9. [配置与编译](#配置与编译)
10. [装饰器](#装饰器)
11. [命名空间](#命名空间)
12. [类型声明文件](#类型声明文件)
13. [工具类型](#工具类型)
14. [实战应用](#实战应用)

## TypeScript 简介

TypeScript 是 JavaScript 的超集，它添加了可选的静态类型和基于类的面向对象编程。

### 为什么选择 TypeScript？

1. **类型安全**：在编译时捕获错误
2. **更好的开发体验**：提供代码补全和重构
3. **面向对象特性**：支持类、接口等
4. **更好的可维护性**：类型系统作为文档

### 安装与使用

```bash
# 安装 TypeScript
npm install -g typescript

# 创建 tsconfig.json
tsc --init

# 编译 TypeScript 文件
tsc app.ts
```

## 基础类型

### 基本类型声明

```typescript
// 基本类型
let isDone: boolean = false;
let decimal: number = 6;
let color: string = "blue";
let list: number[] = [1, 2, 3];
let tuple: [string, number] = ["hello", 10];

// 枚举
enum Color {
  Red,
  Green,
  Blue
}
let c: Color = Color.Green;

// any 类型
let notSure: any = 4;
notSure = "maybe a string instead";

// void 类型
function warnUser(): void {
  console.log("This is a warning message");
}

// null 和 undefined
let u: undefined = undefined;
let n: null = null;
```

### 类型断言

```typescript
// 方式一：尖括号语法
let someValue: any = "this is a string";
let strLength: number = (<string>someValue).length;

// 方式二：as 语法
let someValue: any = "this is a string";
let strLength: number = (someValue as string).length;
```

## 接口与类型

### 接口定义

```typescript
interface User {
  name: string;
  age: number;
  email?: string;  // 可选属性
  readonly id: number;  // 只读属性
}

// 实现接口
const user: User = {
  name: "张三",
  age: 25,
  id: 1
};

// 接口继承
interface Employee extends User {
  department: string;
  salary: number;
}
```

### 类型别名

```typescript
type Point = {
  x: number;
  y: number;
};

type ID = string | number;

type Callback = (data: string) => void;
```

## 函数

### 函数类型

```typescript
// 函数类型声明
function add(x: number, y: number): number {
  return x + y;
}

// 可选参数和默认参数
function buildName(firstName: string, lastName?: string) {
  return lastName ? `${firstName} ${lastName}` : firstName;
}

// 剩余参数
function sum(...numbers: number[]): number {
  return numbers.reduce((a, b) => a + b, 0);
}

// 函数类型接口
interface SearchFunc {
  (source: string, subString: string): boolean;
}
```

## 类

### 类的基本使用

```typescript
class Animal {
  private name: string;
  protected age: number;
  
  constructor(name: string, age: number) {
    this.name = name;
    this.age = age;
  }
  
  public makeSound(): void {
    console.log("Some sound");
  }
}

// 继承
class Dog extends Animal {
  constructor(name: string, age: number) {
    super(name, age);
  }
  
  public makeSound(): void {
    console.log("Woof!");
  }
}
```

### 访问修饰符

- `public`：默认，可以在任何地方访问
- `private`：只能在类内部访问
- `protected`：可以在类内部和子类中访问

## 泛型

### 泛型基础

```typescript
// 泛型函数
function identity<T>(arg: T): T {
  return arg;
}

// 泛型接口
interface GenericIdentityFn<T> {
  (arg: T): T;
}

// 泛型类
class GenericNumber<T> {
  zeroValue: T;
  add: (x: T, y: T) => T;
}
```

### 泛型约束

```typescript
interface Lengthwise {
  length: number;
}

function loggingIdentity<T extends Lengthwise>(arg: T): T {
  console.log(arg.length);
  return arg;
}
```

## 高级类型

### 交叉类型

```typescript
type Combined = Type1 & Type2;
```

### 联合类型

```typescript
type StringOrNumber = string | number;
```

### 类型守卫

```typescript
function isString(value: any): value is string {
  return typeof value === "string";
}
```

## 模块

### 导出

```typescript
// 命名导出
export interface User {
  name: string;
}

export function getUser(): User {
  return { name: "John" };
}

// 默认导出
export default class UserService {
  // ...
}
```

### 导入

```typescript
import { User, getUser } from './user';
import UserService from './UserService';
```

## 配置与编译

### tsconfig.json 重要配置

```json
{
  "compilerOptions": {
    "target": "es5",
    "module": "commonjs",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  }
}
```

### 编译选项说明

- `target`：指定 ECMAScript 目标版本
- `module`：指定模块代码生成方式
- `strict`：启用所有严格类型检查选项
- `esModuleInterop`：启用 ES 模块互操作性
- `skipLibCheck`：跳过声明文件的类型检查
- `forceConsistentCasingInFileNames`：强制文件名大小写一致

## 装饰器

### 装饰器基础

装饰器是一种特殊类型的声明，可以被附加到类声明、方法、属性或参数上。

```typescript
// 类装饰器
function log(target: any) {
  console.log('类被装饰');
}

@log
class Example {
  // ...
}

// 方法装饰器
function enumerable(value: boolean) {
  return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    descriptor.enumerable = value;
  };
}

class Example2 {
  @enumerable(false)
  method() {}
}

// 属性装饰器
function required(target: any, propertyKey: string) {
  let value = target[propertyKey];
  
  const getter = function() {
    return value;
  };
  
  const setter = function(newVal: any) {
    if (newVal === undefined) {
      throw new Error('属性不能为空');
    }
    value = newVal;
  };
  
  Object.defineProperty(target, propertyKey, {
    get: getter,
    set: setter
  });
}

class Example3 {
  @required
  name: string;
}
```

### 装饰器工厂

```typescript
function logWithPrefix(prefix: string) {
  return function(target: any) {
    console.log(`${prefix} - 类被装饰`);
  };
}

@logWithPrefix('DEBUG')
class Example4 {
  // ...
}
```

## 命名空间

### 基本使用

```typescript
namespace Validation {
  export interface StringValidator {
    isValid(s: string): boolean;
  }
  
  export class LettersOnlyValidator implements StringValidator {
    isValid(s: string): boolean {
      return /^[A-Za-z]+$/.test(s);
    }
  }
  
  export class ZipCodeValidator implements StringValidator {
    isValid(s: string): boolean {
      return /^\d{5}$/.test(s);
    }
  }
}

// 使用命名空间
let validators: { [s: string]: Validation.StringValidator } = {};
validators["Letters only"] = new Validation.LettersOnlyValidator();
validators["Zip Code"] = new Validation.ZipCodeValidator();
```

### 命名空间合并

```typescript
namespace Animals {
  export class Zebra { }
}

namespace Animals {
  export interface Legged { numberOfLegs: number; }
  export class Dog { }
}
```

## 类型声明文件

### 声明文件基础

```typescript
// 声明文件示例 (example.d.ts)
declare module 'example' {
  export interface Config {
    name: string;
    version: string;
  }
  
  export function init(config: Config): void;
  export function getVersion(): string;
}
```

### 模块声明

```typescript
// 模块声明
declare module '*.css' {
  const content: { [className: string]: string };
  export default content;
}

declare module '*.svg' {
  const content: any;
  export default content;
}
```

### 全局声明

```typescript
// 全局声明
declare global {
  interface Window {
    myGlobal: string;
  }
  
  function myGlobalFunction(): void;
}
```

## 工具类型

### 内置工具类型

```typescript
// Partial<T>
interface Todo {
  title: string;
  description: string;
}

type PartialTodo = Partial<Todo>;
// 等价于：
// {
//   title?: string;
//   description?: string;
// }

// Required<T>
type RequiredTodo = Required<PartialTodo>;
// 等价于 Todo

// Readonly<T>
type ReadonlyTodo = Readonly<Todo>;
// 所有属性变为只读

// Pick<T, K>
type TodoPreview = Pick<Todo, "title">;
// 只包含 title 属性

// Omit<T, K>
type TodoWithoutDescription = Omit<Todo, "description">;
// 排除 description 属性

// Record<K, T>
type CatInfo = {
  age: number;
  breed: string;
}
type CatName = "miffy" | "boris";
const cats: Record<CatName, CatInfo> = {
  miffy: { age: 10, breed: "Persian" },
  boris: { age: 5, breed: "Maine Coon" }
};
```

### 自定义工具类型

```typescript
// 将类型的所有属性变为可选
type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

// 将类型的所有属性变为只读
type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P];
};

// 将类型的所有属性变为必需
type DeepRequired<T> = {
  [P in keyof T]-?: T[P] extends object ? DeepRequired<T[P]> : T[P];
};
```

## 实战应用

### React 组件示例

```typescript
import React, { useState, useEffect } from 'react';

interface User {
  id: number;
  name: string;
  email: string;
}

interface UserListProps {
  users: User[];
  onUserSelect: (user: User) => void;
}

const UserList: React.FC<UserListProps> = ({ users, onUserSelect }) => {
  const [filteredUsers, setFilteredUsers] = useState<User[]>([]);
  const [searchTerm, setSearchTerm] = useState<string>('');

  useEffect(() => {
    const filtered = users.filter(user =>
      user.name.toLowerCase().includes(searchTerm.toLowerCase())
    );
    setFilteredUsers(filtered);
  }, [users, searchTerm]);

  return (
    <div>
      <input
        type="text"
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        placeholder="搜索用户..."
      />
      <ul>
        {filteredUsers.map(user => (
          <li key={user.id} onClick={() => onUserSelect(user)}>
            {user.name} - {user.email}
          </li>
        ))}
      </ul>
    </div>
  );
};
```

### API 调用示例

```typescript
interface ApiResponse<T> {
  data: T;
  status: number;
  message: string;
}

interface User {
  id: number;
  name: string;
  email: string;
}

async function fetchUser(id: number): Promise<ApiResponse<User>> {
  try {
    const response = await fetch(`/api/users/${id}`);
    const data = await response.json();
    return {
      data,
      status: response.status,
      message: 'Success'
    };
  } catch (error) {
    throw new Error('Failed to fetch user');
  }
}

// 使用示例
async function displayUser(id: number) {
  try {
    const result = await fetchUser(id);
    console.log(result.data.name);
  } catch (error) {
    console.error(error);
  }
}
```

### 状态管理示例

```typescript
// 使用 TypeScript 实现简单的状态管理
type Action<T> = {
  type: string;
  payload: T;
};

type Reducer<S, A> = (state: S, action: A) => S;

class Store<S, A extends Action<any>> {
  private state: S;
  private reducer: Reducer<S, A>;
  private listeners: Array<() => void> = [];

  constructor(reducer: Reducer<S, A>, initialState: S) {
    this.reducer = reducer;
    this.state = initialState;
  }

  getState(): S {
    return this.state;
  }

  dispatch(action: A): void {
    this.state = this.reducer(this.state, action);
    this.listeners.forEach(listener => listener());
  }

  subscribe(listener: () => void): () => void {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }
}

// 使用示例
interface TodoState {
  todos: string[];
}

type TodoAction = 
  | { type: 'ADD_TODO'; payload: string }
  | { type: 'REMOVE_TODO'; payload: number };

const todoReducer: Reducer<TodoState, TodoAction> = (state, action) => {
  switch (action.type) {
    case 'ADD_TODO':
      return { ...state, todos: [...state.todos, action.payload] };
    case 'REMOVE_TODO':
      return {
        ...state,
        todos: state.todos.filter((_, index) => index !== action.payload)
      };
    default:
      return state;
  }
};

const store = new Store(todoReducer, { todos: [] });
```

## 高级类型操作

### 类型操作符

```typescript
// keyof 操作符
interface Person {
  name: string;
  age: number;
}

type PersonKeys = keyof Person; // "name" | "age"

// typeof 操作符
const colors = ['red', 'green', 'blue'];
type Colors = typeof colors; // string[]

// 索引访问类型
type NameType = Person['name']; // string
type AgeType = Person['age']; // number

// 条件类型
type IsString<T> = T extends string ? true : false;
type A = IsString<'hello'>; // true
type B = IsString<123>; // false
```

### 类型推断与约束

```typescript
// 类型推断
type Unpacked<T> = 
  T extends (infer U)[] ? U :
  T extends (...args: any[]) => infer U ? U :
  T extends Promise<infer U> ? U :
  T;

// 使用示例
type T0 = Unpacked<string[]>; // string
type T1 = Unpacked<Promise<string>>; // string
type T2 = Unpacked<() => number>; // number

// 类型约束
type NonNullable<T> = T extends null | undefined ? never : T;
type T3 = NonNullable<string | null | undefined>; // string
```

## 函数类型进阶

### 函数重载

```typescript
// 函数重载声明
function process(x: string): string;
function process(x: number): number;
function process(x: string | number): string | number {
  if (typeof x === 'string') {
    return x.toUpperCase();
  } else {
    return x * 2;
  }
}

// 方法重载
class Calculator {
  add(x: number, y: number): number;
  add(x: string, y: string): string;
  add(x: any, y: any): any {
    return x + y;
  }
}
```

### 函数类型操作

```typescript
// 函数参数类型
type Parameters<T extends (...args: any) => any> = T extends (...args: infer P) => any ? P : never;

// 函数返回值类型
type ReturnType<T extends (...args: any) => any> = T extends (...args: any) => infer R ? R : any;

// 函数 this 类型
interface ThisType<T> {
  (this: T, ...args: any[]): any;
}

// 使用示例
function getThisType<T>(fn: ThisType<T>): T {
  return fn.call({} as T);
}
```

## 高级接口特性

### 接口合并

```typescript
// 接口合并
interface Box {
  height: number;
  width: number;
}

interface Box {
  scale: number;
}

// 结果等同于：
// interface Box {
//   height: number;
//   width: number;
//   scale: number;
// }

// 命名空间与接口合并
namespace Validation {
  export interface StringValidator {
    isValid(s: string): boolean;
  }
}

interface Validation.StringValidator {
  validate(s: string): boolean;
}
```

### 接口扩展

```typescript
// 接口扩展
interface Animal {
  name: string;
}

interface Dog extends Animal {
  breed: string;
}

interface WorkingDog extends Dog {
  job: string;
}

// 多接口扩展
interface A {
  a: string;
}

interface B {
  b: number;
}

interface C extends A, B {
  c: boolean;
}
```

## 类型系统进阶

### 类型保护

```typescript
// 类型谓词
function isString(value: any): value is string {
  return typeof value === 'string';
}

// instanceof 类型保护
class Animal {
  name: string;
  constructor(name: string) {
    this.name = name;
  }
}

class Dog extends Animal {
  breed: string;
  constructor(name: string, breed: string) {
    super(name);
    this.breed = breed;
  }
}

function isDog(animal: Animal): animal is Dog {
  return animal instanceof Dog;
}

// 使用示例
function processAnimal(animal: Animal) {
  if (isDog(animal)) {
    console.log(animal.breed); // 类型被收窄为 Dog
  }
}
```

### 类型断言

```typescript
// 类型断言
const value = 'hello' as const;
type ValueType = typeof value; // "hello"

// 双重断言
const value2 = 'hello' as any as number;

// const 断言
const colors = ['red', 'green', 'blue'] as const;
type Colors = typeof colors[number]; // "red" | "green" | "blue"
```

## 模块系统进阶

### 模块解析

```typescript
// 模块解析策略
// tsconfig.json
{
  "compilerOptions": {
    "moduleResolution": "node",
    "baseUrl": "./src",
    "paths": {
      "@/*": ["*"]
    }
  }
}

// 使用示例
import { Component } from '@/components/Component';
```

### 模块声明

```typescript
// 模块声明
declare module '*.css' {
  const content: { [className: string]: string };
  export default content;
}

declare module '*.svg' {
  const content: any;
  export default content;
}

// 模块扩展
declare module 'express' {
  interface Request {
    user?: any;
  }
}
```

## 实用工具类型

### 类型操作工具

```typescript
// 排除类型
type Exclude<T, U> = T extends U ? never : T;
type T0 = Exclude<"a" | "b" | "c", "a">; // "b" | "c"

// 提取类型
type Extract<T, U> = T extends U ? T : never;
type T1 = Extract<"a" | "b" | "c", "a" | "f">; // "a"

// 不可空类型
type NonNullable<T> = T extends null | undefined ? never : T;
type T2 = NonNullable<string | null | undefined>; // string

// 构造函数类型
type Constructor<T> = new (...args: any[]) => T;
type T3 = Constructor<{ name: string }>;
```

### 类型转换工具

```typescript
// 类型转换
type ToString<T> = T extends string ? T : never;
type T4 = ToString<"hello" | 123>; // "hello"

// 类型映射
type MapToPromise<T> = { [K in keyof T]: Promise<T[K]> };
type T5 = MapToPromise<{ name: string; age: number }>;
// { name: Promise<string>; age: Promise<number> }

// 类型过滤
type Filter<T, U> = T extends U ? T : never;
type T6 = Filter<"a" | "b" | "c", "a">; // "a"
```

## 性能优化进阶

### 类型优化

```typescript
// 使用 const 断言优化类型推断
const colors = ['red', 'green', 'blue'] as const;
type Color = typeof colors[number];

// 使用 readonly 优化不可变数据
interface Config {
  readonly apiKey: string;
  readonly endpoint: string;
}

// 使用类型别名优化复杂类型
type DeepReadonly<T> = {
  readonly [P in keyof T]: DeepReadonly<T[P]>;
};
```

### 编译优化

```json
{
  "compilerOptions": {
    "incremental": true,
    "tsBuildInfoFile": "./build/.tsbuildinfo",
    "isolatedModules": true,
    "noEmit": true,
    "skipLibCheck": true,
    "preserveConstEnums": true,
    "removeComments": true,
    "sourceMap": true,
    "declaration": true,
    "declarationMap": true
  }
}
```

## 调试与测试

### 类型调试

```typescript
// 类型调试工具
type Debug<T> = {
  [K in keyof T]: T[K];
};

// 类型检查
type IsEqual<T, U> = T extends U ? U extends T ? true : false : false;
type T7 = IsEqual<string, string>; // true
type T8 = IsEqual<string, number>; // false

// 类型打印
type Print<T> = {
  [K in keyof T]: T[K];
} & { __type: T };
```

### 测试工具

```typescript
// 类型测试
type Expect<T extends true> = T;
type ExpectTrue<T extends true> = T;
type ExpectFalse<T extends false> = T;

// 类型相等测试
type Equal<X, Y> = (<T>() => T extends X ? 1 : 2) extends (<T>() => T extends Y ? 1 : 2) ? true : false;

// 使用示例
type Test1 = Expect<Equal<string, string>>; // true
type Test2 = Expect<Equal<string, number>>; // 类型错误
```

## 最佳实践总结

### 类型定义最佳实践

1. **使用接口定义对象结构**
   ```typescript
   interface User {
     name: string;
     age: number;
   }
   ```

2. **使用类型别名定义联合类型**
   ```typescript
   type Status = 'pending' | 'success' | 'error';
   ```

3. **使用泛型增加代码复用性**
   ```typescript
   function identity<T>(arg: T): T {
     return arg;
   }
   ```

### 代码组织最佳实践

1. **使用命名空间组织代码**
   ```typescript
   namespace Utils {
     export function formatDate(date: Date): string {
       return date.toISOString();
     }
   }
   ```

2. **使用模块组织代码**
   ```typescript
   // user.ts
   export interface User {
     name: string;
   }
   
   // userService.ts
   import { User } from './user';
   export class UserService {
     getUser(): User {
       return { name: 'John' };
     }
   }
   ```

### 性能优化最佳实践

1. **使用 const 断言**
   ```typescript
   const colors = ['red', 'green', 'blue'] as const;
   ```

2. **使用 readonly 类型**
   ```typescript
   interface Config {
     readonly apiKey: string;
   }
   ```

3. **使用类型推断**
   ```typescript
   const user = { name: 'John', age: 30 };
   type User = typeof user;
   ```

## 常见问题解决方案

### 类型错误处理

```typescript
// 类型错误处理
type ErrorType = {
  message: string;
  code: number;
};

function handleError(error: unknown): ErrorType {
  if (error instanceof Error) {
    return {
      message: error.message,
      code: 500
    };
  }
  return {
    message: 'Unknown error',
    code: 500
  };
}
```

### 类型兼容性处理

```typescript
// 类型兼容性
interface Animal {
  name: string;
}

interface Dog extends Animal {
  breed: string;
}

function processAnimal(animal: Animal) {
  // 类型兼容性检查
  if ('breed' in animal) {
    console.log(animal.breed);
  }
}
```

