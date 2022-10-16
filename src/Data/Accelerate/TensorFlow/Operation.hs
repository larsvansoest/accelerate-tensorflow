{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies      #-}

module Data.Accelerate.TensorFlow.Operation where

import Data.Array.Accelerate.AST.Exp
import Data.Array.Accelerate.AST.Operation
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Array.Accelerate.Analysis.Hash.Exp
import Data.Array.Accelerate.Analysis.Hash.Operation
import Data.Array.Accelerate.Backend
import Data.Array.Accelerate.Trafo.Partitioning.ILP.Graph hiding (Var)
import Data.Array.Accelerate.Trafo.Partitioning.ILP.Labels
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Representation.Type

data TensorOp op where
  TConst :: ScalarType s -> s -> TensorOp ()
  TPrimFun :: PrimFun (a -> b) -> TensorOp (In sh a -> Out sh b -> ())

instance PrettyOp TensorOp where
  prettyOp (TConst _ _) = "TTensor"
  prettyOp (TPrimFun _) = "TBinOp"

instance NFData' TensorOp where
  rnf' !_ = ()

mapXTimesTwoPlusOne :: Arg env (In sh s) -> Arg env (Out sh t) -> OperationAcc TensorOp env ()
mapXTimesTwoPlusOne (ArgArray _ (ArrayR shIn _) gvIn gvbIn) (ArgArray _ (ArrayR shOut _) gvOut gvbOut)
  = let
      tInt64 :: NumType Int64
      tInt64 = IntegralNumType TypeInt64

      timesOp = TPrimFun (PrimMul tInt64)
      plusOp  = TPrimFun (PrimAdd tInt64) -- Todo: include

      tNewIn :: TupR ScalarType (Int64, Int64)
      tNewIn = TupRpair 
        (TupRsingle (SingleScalarType (NumSingleType tInt64))) 
        (TupRsingle (SingleScalarType (NumSingleType tInt64)))

      newGvbIn :: GroundVars env (Buffers (Int64, Int64))
      newGvbIn = undefined

      tNewOut :: TypeR Int64
      tNewOut = undefined

      newGvbOut :: TupR (Var GroundR env) (Buffer Int64)
      newGvbOut = undefined
      
    in Exec timesOp (
      ArgArray In (ArrayR shIn tNewIn) gvIn newGvbIn 
      :>: ArgArray Out (ArrayR shOut tNewOut) gvOut newGvbOut
      :>: ArgsNil
    )

mkMapF :: Arg env (Fun' (s -> t)) -> Arg env (In sh s) -> Arg env (Out sh t) -> OperationAcc TensorOp env ()
-- Constant values
mkMapF (ArgFun (Body (Const s e)))   _ _ = Exec (TConst s e) ArgsNil
mkMapF (ArgFun (Body (PrimConst _))) _ _ = undefined
-- Primitive scalar operations
mkMapF (ArgFun (Body (PrimApp f exp))) (ArgArray m (ArrayR sh tR) g b) o 
  = Exec (TPrimFun f) (ArgArray m (ArrayR sh _) g _ :>: _ :>: ArgsNil)

mkMapF (ArgFun (Body (ArrayInstr _ _))) input output = undefined

-- Others
mkMapF (ArgFun (Body (Let _ _ _))) input output = undefined
mkMapF (ArgFun (Body (Evar _))) input output = undefined
mkMapF (ArgFun (Body (Foreign _ _ _ _))) input output = undefined
mkMapF (ArgFun (Body (VecUnpack _ _))) input output = undefined
mkMapF (ArgFun (Body (IndexSlice _ _ _))) input output = undefined
mkMapF (ArgFun (Body (IndexFull _ _ _))) input output = undefined
mkMapF (ArgFun (Body (FromIndex _ _ _))) input output = undefined
mkMapF (ArgFun (Body (Case _ _ _))) input output = undefined
mkMapF (ArgFun (Body (Cond _ _ _))) input output = undefined
mkMapF (ArgFun (Body (While _ _ _))) input output = undefined
mkMapF (ArgFun (Body (Undef _))) input output = undefined
mkMapF (ArgFun (Body (Coerce _ _ _))) input output = undefined
mkMapF (ArgFun (Lam (LeftHandSideSingle (SingleScalarType (NumSingleType (IntegralNumType _)))) f)) input output = undefined
mkMapF (ArgFun (Lam (LeftHandSideSingle (SingleScalarType (NumSingleType (FloatingNumType _)))) f)) input output = undefined
mkMapF (ArgFun (Lam (LeftHandSideSingle (VectorScalarType _)) f)) input output = undefined
mkMapF (ArgFun (Lam (LeftHandSideWildcard _) f)) input output = undefined
mkMapF (ArgFun (Lam (LeftHandSidePair _ _) f)) input output = undefined

-- Redundant pattern matches
mkMapF (ArgFun (Body (Pair _ _))) input output = undefined
mkMapF (ArgFun (Body Nil)) input output = undefined
mkMapF (ArgFun (Body (VecPack _ _))) input output = undefined
mkMapF (ArgFun (Body (ShapeSize _ _))) input output = undefined
mkMapF (ArgFun (Body (ToIndex _ _ _))) input output = undefined

instance DesugarAcc TensorOp where
  mkMap = mkMapF
  mkGenerate = undefined
  mkPermute = undefined
