{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE InstanceSigs #-}

module Data.Accelerate.TensorFlow.Operation where

import Data.Accelerate.TensorFlow.Type
    ( TFOrd, OneOf, TFNum, TFAll, TFFloat, TensorType, TFInt, TFNum', TFMod )
import Data.Array.Accelerate.AST.Operation
    ( PrimBool, Mut, Out, In, NFData'(..), Var' )
import Data.Array.Accelerate
    ( MakesILP(..),
      ShrinkArg(..),
      SLVOperation(..),
      SimplifyOperation,
      PrettyOp(prettyOp),
      EncodeOperation(..))
import Data.Array.Accelerate.Pretty.Exp ( Adoc )
import Data.Array.Accelerate.Trafo.Partitioning.ILP.Labels (LabelledArg(..))
import Data.Array.Accelerate.Trafo.Partitioning.ILP.Graph
    ( LabelledArgOp(LOp) )
import Data.Array.Accelerate.Analysis.Hash.Exp (intHost, hashQ, encodeScalarConst, Builder, encodeScalarType)
import Data.Array.Accelerate.Representation.Shape (DIM1)
import Data.Array.Accelerate.Type (ScalarType)

data TensorOp op where
  TConstant    :: OneOf TFAll a => ScalarType a -> a -> TensorOp (Out sh a -> ())
  TVar         :: OneOf TFAll a => ScalarType a -> TensorOp (Var' a -> Out sh a -> ())
  TId          :: OneOf TFAll a => TensorOp (In sh a -> Out sh a -> ())
  TSelect      :: OneOf TFAll a => TensorOp (In sh PrimBool -> In sh a -> In sh a -> Out sh a -> ())
  TGather      :: OneOf TFAll a => TensorOp (In DIM1 a -> In sh Int -> Out sh a -> ())
  TWhere       :: TensorOp (In DIM1 Int -> Out DIM1 Int -> ())
  TBooleanMask :: OneOf TFAll a => TensorOp (In DIM1 a -> In DIM1 PrimBool -> Out DIM1 a -> ())
  TCast :: (TensorType a, TensorType b) => TensorOp (In sh a -> Out sh b -> ())

  -- operators from Num
  TAdd  :: OneOf TFNum  a => TensorOp (In sh a -> In sh a -> Out sh a -> ())
  TMul  :: OneOf TFNum  a => TensorOp (In sh a -> In sh a -> Out sh a -> ())
  TSub  :: OneOf TFNum  a => TensorOp (In sh a -> In sh a -> Out sh a -> ())
  TNeg  :: OneOf TFNum' a => TensorOp (In sh a -> Out sh a -> ())
  TAbs  :: OneOf TFNum' a => TensorOp (In sh a -> Out sh a -> ())
  TSign :: OneOf TFNum' a => TensorOp (In sh a -> Out sh a -> ())

  -- operators from Integral
  TTruncateDiv :: OneOf TFNum a => TensorOp (In sh a -> In sh a -> Out sh a -> ())
  TTruncateMod :: OneOf TFMod a => TensorOp (In sh a -> In sh a -> Out sh a -> ())
  TRealDiv     :: OneOf TFNum a => TensorOp (In sh a -> In sh a -> Out sh a -> ())

  -- operators from Bits & FiniteBits
  TBitwiseAnd :: OneOf TFInt a => TensorOp (In sh a -> In sh a -> Out sh a -> ())
  TBitwiseOr  :: OneOf TFInt a => TensorOp (In sh a -> In sh a -> Out sh a -> ())
  TBitwiseXor :: OneOf TFInt a => TensorOp (In sh a -> In sh a -> Out sh a -> ())
  TInvert     :: OneOf TFInt a => TensorOp (In sh a -> Out sh a -> ())

  -- operators from Fractional and Floating
  TReciprocal :: OneOf TFFloat a => TensorOp (In sh a -> Out sh a -> ())
  TSin        :: OneOf TFFloat a => TensorOp (In sh a -> Out sh a -> ())
  TCos        :: OneOf TFFloat a => TensorOp (In sh a -> Out sh a -> ())
  TTan        :: OneOf TFFloat a => TensorOp (In sh a -> Out sh a -> ())
  TAsin       :: OneOf TFFloat a => TensorOp (In sh a -> Out sh a -> ())
  TAcos       :: OneOf TFFloat a => TensorOp (In sh a -> Out sh a -> ())
  TAtan       :: OneOf TFFloat a => TensorOp (In sh a -> Out sh a -> ())
  TSinh       :: OneOf TFFloat a => TensorOp (In sh a -> Out sh a -> ())
  TCosh       :: OneOf TFFloat a => TensorOp (In sh a -> Out sh a -> ())
  TTanh       :: OneOf TFFloat a => TensorOp (In sh a -> Out sh a -> ())
  TAsinh      :: OneOf TFFloat a => TensorOp (In sh a -> Out sh a -> ())
  TAcosh      :: OneOf TFFloat a => TensorOp (In sh a -> Out sh a -> ())
  TAtanh      :: OneOf TFFloat a => TensorOp (In sh a -> Out sh a -> ())
  TSqrt       :: OneOf TFFloat a => TensorOp (In sh a -> Out sh a -> ())
  TExp        :: OneOf TFFloat a => TensorOp (In sh a -> Out sh a -> ())
  TLog        :: OneOf TFFloat a => TensorOp (In sh a -> Out sh a -> ())
  TPow        :: OneOf TFFloat a => TensorOp (In sh a -> In sh a -> Out sh a -> ())
  TLog1p      :: OneOf TFFloat a => TensorOp (In sh a -> Out sh a -> ())
  TAtan2      :: OneOf TFFloat a => TensorOp (In sh a -> In sh a -> Out sh a -> ())

  -- relational and equality operators
  TLess         :: OneOf TFOrd a => TensorOp (In sh a -> In sh a -> Out sh PrimBool -> ())
  TGreater      :: OneOf TFOrd a => TensorOp (In sh a -> In sh a -> Out sh PrimBool -> ())
  TLessEqual    :: OneOf TFOrd a => TensorOp (In sh a -> In sh a -> Out sh PrimBool -> ())
  TGreaterEqual :: OneOf TFOrd a => TensorOp (In sh a -> In sh a -> Out sh PrimBool -> ())
  TEqual        :: OneOf TFOrd a => TensorOp (In sh a -> In sh a -> Out sh PrimBool -> ())
  TNotEqual     :: OneOf TFOrd a => TensorOp (In sh a -> In sh a -> Out sh PrimBool -> ())
  TMaximum      :: OneOf TFOrd a => TensorOp (In sh a -> In sh a -> Out sh a -> ())
  TMinimum      :: OneOf TFOrd a => TensorOp (In sh a -> In sh a -> Out sh a -> ())

  -- logical operators
  TLogicalAnd :: TensorOp (In sh PrimBool -> In sh PrimBool -> Out sh PrimBool -> ())
  TLogicalOr  :: TensorOp (In sh PrimBool -> In sh PrimBool -> Out sh PrimBool -> ())
  TLogicalNot :: TensorOp (In sh PrimBool -> Out sh PrimBool -> ())

  TTensorScatter :: ScatterFun -> TensorOp (Mut sh' s -> In sh sh' -> In sh s -> ())

instance EncodeOperation TensorOp where
  encodeOperation :: TensorOp t -> Builder
  encodeOperation t = case t of
    TConstant st s -> intHost $(hashQ ("Constant" :: String)) <> encodeScalarConst st s
    TVar st -> intHost $(hashQ ("Var" :: String)) <> encodeScalarType st
    TWhere -> intHost $(hashQ ("Where" :: String))
    TId -> intHost $(hashQ ("Id" :: String))
    TSelect -> intHost $(hashQ ("Select" :: String))
    TAdd -> intHost $(hashQ ("Add" :: String))
    TMul -> intHost $(hashQ ("Mul" :: String))
    TSub -> intHost $(hashQ ("Sub" :: String))
    TNeg -> intHost $(hashQ ("Neg" :: String))
    TAbs -> intHost $(hashQ ("Abs" :: String))
    TSign -> intHost $(hashQ ("Sign" :: String))
    TTruncateDiv -> intHost $(hashQ ("TruncateDiv" :: String))
    TTruncateMod -> intHost $(hashQ ("TruncateMod" :: String))
    TTensorScatter sf -> encodeScatterFun sf
    TBooleanMask -> intHost $(hashQ ("BooleanMask" :: String))
    TReciprocal -> intHost $(hashQ ("Reciprocal" :: String))
    TSin -> intHost $(hashQ ("Sin" :: String))
    TCos -> intHost $(hashQ ("Cos" :: String))
    TTan -> intHost $(hashQ ("Tan" :: String))
    TAsin -> intHost $(hashQ ("Asin" :: String))
    TAcos -> intHost $(hashQ ("Acos" :: String))
    TAtan -> intHost $(hashQ ("Atan" :: String))
    TSinh -> intHost $(hashQ ("Sinh" :: String))
    TCosh -> intHost $(hashQ ("Cosh" :: String))
    TTanh -> intHost $(hashQ ("Tanh" :: String))
    TAsinh -> intHost $(hashQ ("Asinh" :: String))
    TAcosh -> intHost $(hashQ ("Acosh" :: String))
    TAtanh -> intHost $(hashQ ("Atanh" :: String))
    TSqrt -> intHost $(hashQ ("Sqrt" :: String))
    TExp -> intHost $(hashQ ("Exp" :: String))
    TLog -> intHost $(hashQ ("Log" :: String))
    TPow -> intHost $(hashQ ("Pow" :: String))
    TLog1p -> intHost $(hashQ ("Log1p" :: String))
    TAtan2 -> intHost $(hashQ ("Atan2" :: String))
    TLess -> intHost $(hashQ ("Less" :: String))
    TGreater -> intHost $(hashQ ("Greater" :: String))
    TLessEqual -> intHost $(hashQ ("LessEqual" :: String))
    TGreaterEqual -> intHost $(hashQ ("GreaterEqual" :: String))
    TEqual -> intHost $(hashQ ("Equal" :: String))
    TNotEqual -> intHost $(hashQ ("NotEqual" :: String))
    TMaximum -> intHost $(hashQ ("Maximum" :: String))
    TMinimum -> intHost $(hashQ ("Minimum" :: String))
    TLogicalAnd -> intHost $(hashQ ("LogicalAnd" :: String))
    TLogicalOr -> intHost $(hashQ ("LogicalOr" :: String))
    TLogicalNot -> intHost $(hashQ ("LogicalNot" :: String))
    TRealDiv -> intHost $(hashQ ("RealDiv" :: String))
    TBitwiseAnd -> intHost $(hashQ ("BitwiseAnd" :: String))
    TBitwiseOr -> intHost $(hashQ ("BitwiseOr" :: String))
    TBitwiseXor -> intHost $(hashQ ("BitwiseXor" :: String))
    TInvert -> intHost $(hashQ ("Invert" :: String))
    TCast -> intHost $(hashQ ("Cast" :: String))
    TGather -> intHost $(hashQ ("Gather" :: String))

instance PrettyOp TensorOp where
  prettyOp :: TensorOp t -> Adoc
  prettyOp _ = "TensorOp"

instance NFData' TensorOp where
  rnf' !_ = ()

instance SimplifyOperation TensorOp where

instance SLVOperation TensorOp where
  slvOperation _ = Nothing

instance ShrinkArg (BackendClusterArg TensorOp) where
  shrinkArg _ NoFusionArg = NoFusionArg
  deadArg NoFusionArg = NoFusionArg

instance NFData' (BackendClusterArg TensorOp) where
  rnf' NoFusionArg = ()

instance MakesILP TensorOp where
  type BackendVar TensorOp = ()
  type BackendArg TensorOp = ()
  data BackendClusterArg TensorOp arg = NoFusionArg
  mkGraph _ _ _ = mempty
  labelLabelledArg _ _ (L arg l) = LOp arg l ()
  getClusterArg (LOp _ _ ()) = NoFusionArg
  finalize = mempty
  encodeBackendClusterArg NoFusionArg = intHost $(hashQ ("NoFusionArg" :: String))

data ScatterFun where
  ScatterFunAdd :: ScatterFun
  ScatterFunMin :: ScatterFun
  deriving Show

encodeScatterFun :: ScatterFun -> Builder
encodeScatterFun ScatterFunAdd = intHost $(hashQ ("ScatterFunAdd" :: String))
encodeScatterFun ScatterFunMin = intHost $(hashQ ("ScatterFunMin" :: String))

prettyScatterFun :: ScatterFun -> Adoc
prettyScatterFun ScatterFunAdd = "ScatterFunAdd"
prettyScatterFun ScatterFunMin = "ScatterFunMin"